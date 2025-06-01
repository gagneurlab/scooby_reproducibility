import os
import yaml
import argparse

import numpy as np
import pandas as pd
import polars as pl
import anndata as ad
import scipy
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset

from enformer_pytorch.data import GenomeIntervalDataset, str_to_one_hot
from scooby.utils.transcriptome import Transcriptome
from borzoi_pytorch.config_borzoi import BorzoiConfig

from scooby.utils.transcriptome import Transcriptome
from scooby.utils.utils import undo_squashed_scale, fix_rev_comp_rna, get_pseudobulk_count_pred, get_gene_slice_and_strand
from scooby.modeling import Scooby

parser = argparse.ArgumentParser()
parser.add_argument('-in_path', type=str, required="True")
parser.add_argument('-out_path', type=str, required="True")
parser.add_argument('-config_path', type=str, required="True")
args = parser.parse_args()

in_path = args.in_path
out_path = args.out_path
config_path = args.config_path

# load and chunk
# parse out indices
idx = out_path.split(".")[-2].split("_")[-2:]
start_idx = int(idx[0])
end_idx = int(idx[1])
print(start_idx, end_idx)
print("Load data")
names = list(pd.read_csv(in_path, nrows=2, sep="\t").columns)
dataset = pd.read_csv(in_path, skiprows=start_idx+1, nrows=(end_idx+1)-start_idx, names=names, sep="\t")
print("Loaded data")
print(dataset.iloc[0])

# load config
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
# get config params
replicate = config['replicate']
data_path = config['data_path']
gtf_file = config['gtf_file']
fasta_file = config['fasta_file']
bed_file = config['bed_file']
span = config['span']
pretrained_path = config['pretrained_path']
return_center_bins_only = config['return_center_bins_only']

embedding_path = config['embedding_path']
cell_type_idx_path = config['cell_type_idx_path']

disable_autocast = config.get('disable_autocast', False)

print(pretrained_path)
print(config_path)

cfg = BorzoiConfig.from_pretrained(pretrained_path)
if return_center_bins_only:
    bins = 6144 # how many bins are predicted
    offset = 163840 # how much after sequence start does the prediction start
else:
    bins = 16384 - 32 # how many bins are predicted
    offset = 512 # how much after sequence start does the prediction start
    cfg.return_center_bins_only = False

def get_gene_slice_and_strand(transcriptome, gene, position, span, bins=bins):
    """
    Retrieves the gene slice and strand information from the transcriptome.

    Args:
        transcriptome: The transcriptome object.
        gene (str): The name of the gene.
        position (int): The genomic position.
        span (int): The span of the genomic region.
        sliced (bool, optional): Whether to slice the output. Defaults to True.

    Returns:
        Tuple[torch.Tensor, str]: The gene slice and strand.
    """
    gene_slice = transcriptome.genes[gene].output_slice(
        position, bins * 32, 32, span=span, 
    )  # select right columns
    strand = transcriptome.genes[gene].strand
    return gene_slice, strand

def geneslice_collate_fn(batch):
    """
    Collate function for a PyTorch DataLoader with fixed and variable length tensors.

    Args:
        batch: A list of tuples, where each tuple contains:
            - A fixed-size tensor.
            - A variable-length tensor.

    Returns:
        A tuple containing:
            - A tensor of stacked fixed-size tensors.
            - A list of variable-length tensors.
    """
    fixed_tensors = []
    variable_tensors = []

    for fixed_tensor, variable_tensor in batch:
        fixed_tensors.append(fixed_tensor)
        variable_tensors.append(torch.from_numpy(variable_tensor))

    # Stack the fixed-size tensors
    stacked_fixed_tensors = torch.stack(fixed_tensors)
    
    return stacked_fixed_tensors, variable_tensors

# VariantDataSet
class VariantDataset(Dataset):

    def __init__(self, snp_df, gtf_file, fasta_file, bed_file,
                context_length = 524288, span=span, offset=offset, bins=bins):
        # load gene regions
        gene_regions = pd.read_table(bed_file,names=['Chromosome','Start','End','gene_name','Strand'])
        # extend gene regions (from 200kb to 500kb)
        gene_regions['Start'] = gene_regions['Start'] - 163840
        gene_regions['End'] = gene_regions['End'] + 163840
        # intersect variants with gene regions (of target gene)
        snp_df = gene_regions.merge(snp_df,on=['Chromosome','gene_name'])
        self.snp_df = snp_df.query('Pos >= Start and Pos < End').reset_index(drop=True)
        # build dict to quickly query the gene regions
        self.gene_to_idx = {g:i for i,g in enumerate(gene_regions['gene_name'])}
        # build genome dataset
        self.gene_region_ds = GenomeIntervalDataset(
            bed_file = bed_file,
            fasta_file = fasta_file,
            filter_df_fn = lambda x: x,
            return_seq_indices = False,
            shift_augs = (0,0),
            rc_aug = False,
            return_augs = True,
            context_length = context_length,
            chr_bed_to_fasta_map = {}
        )
        self.transcriptome = Transcriptome(gtf_file)
        self.span = span
        self.offset = offset
        self.bins = bins
        # find genes which the transcriptome does not have
        # or where the associated region offers no bins
        blacklist = []
        for _,rec in self.snp_df.iterrows():
            gene = rec['gene_name']
            if gene not in self.transcriptome.genes.keys():
                blacklist.append(gene)
                continue
            gene_slice, gene_strand = get_gene_slice_and_strand(self.transcriptome, 
                                                            gene, 
                                                            rec['Start'] + self.offset,
                                                            span=self.span)
            if gene_slice.shape[0] == 0:
                blacklist.append(gene)
        # remove these genes
        self.snp_df = self.snp_df.query('gene_name not in @blacklist')

    def __len__(self):
        return len(self.snp_df) * 2

    def __getitem__(self,idx):
        # get variant
        allele = 'Ref' if idx < len(self.snp_df) else 'Alt'
        rec = self.snp_df.iloc[idx % len(self.snp_df)]
        gene = rec['gene_name']
        strand = rec['Strand']
        pos = rec['Pos'] - rec['Start']# - 1
        # get sequence of associated gene
        gene_idx = self.gene_to_idx[gene]
        seq = self.gene_region_ds[gene_idx][0]
        # if ref, compute offset, check nuc is correct
        nuc = str_to_one_hot(rec[allele]).squeeze()
        varlen = len(rec[allele])
        if allele == 'Ref':
            assert torch.allclose(seq[pos:pos+varlen], nuc), gene + ":" + str(seq[pos:pos+varlen])
        # if alt, compute offset, insert variant
        else:
            seq[pos:pos+varlen] = nuc
        # get bins
        gene_slice, gene_strand = get_gene_slice_and_strand(self.transcriptome, 
                                                            gene, 
                                                            rec['Start'] + self.offset,
                                                            span=self.span)
        assert strand == gene_strand, gene
        assert gene_slice.shape[0] > 0, gene # we do not want an empty span
        # make sequence sense
        if strand == '-':
            seq = seq.flip(dims=(0,1))
            gene_slice = (self.bins - 1) - gene_slice
        return seq, gene_slice

# scooby predict function
def predict(model, seqs, seqs_rev_comp, conv_weights, conv_biases, bins_to_predict = None):
    bs = seqs.shape[0]
    assert bs == 1
    with torch.no_grad():
        with torch.autocast("cuda"):
            outputs = model.forward_sequence_w_convs(seqs, conv_weights, conv_biases, bins_to_predict = bins_to_predict)
            if bins_to_predict is not None:
                outputs_rev_comp = model.forward_sequence_w_convs(seqs_rev_comp, conv_weights, conv_biases, bins_to_predict = (6143 - bins_to_predict))
            else:
                outputs_rev_comp = model.forward_sequence_w_convs(seqs_rev_comp, conv_weights, conv_biases, bins_to_predict = None)
    flipped_version = torch.flip(outputs_rev_comp,(1,-3))
    outputs_rev_comp_test = fix_rev_comp_rna(flipped_version)
    return (outputs + outputs_rev_comp_test)/2
        

# Make model
device = 'cuda'
scooby = Scooby.from_pretrained(
    pretrained_path, 
    cell_emb_dim=10,
    embedding_dim=1920,
    n_tracks=2,
    return_center_bins_only=True,
    disable_cache=False,
    use_transform_borzoi_emb=True,
)
scooby.eval()
scooby.to(device)


# Prepare cell stuff
embedding = pd.read_parquet(embedding_path)
cell_type_index = pd.read_parquet(cell_type_idx_path)
cell_type_index['celltype'] = cell_type_index['celltype'].str.replace(' ', '_').str.replace("/", "_")
cell_type_index = cell_type_index.sort_values('celltype')

cell_indices  = []
size_factors_per_ct = []
for _, row in tqdm.tqdm(cell_type_index.iterrows(),disable = True):
    cell_indices.append(
        torch.from_numpy(
            np.vstack(
                embedding.iloc[row['cellindex']]['embedding'].values # gets embeddings of all cells of the cell type
                )
            ).unsqueeze(0)
        ) # prep cell_embeddings
# get conv weights and biases for all cells sorted by cell type in a list
cell_emb_conv_weights_and_biases = []
for cell_emb_idx in tqdm.tqdm(cell_indices, disable = True):
    cell_emb_idx = cell_emb_idx.to(device)
    conv_weights, conv_biases = scooby.forward_cell_embs_only(cell_emb_idx)
    cell_emb_conv_weights_and_biases.append((conv_weights.to(torch.float16), conv_biases.to(torch.float16)))



# Make dl
bs = 1 #8 if 'flashzoi' in pretrained_path else (2 if disable_autocast else 4)
var_ds = VariantDataset(dataset, gtf_file, fasta_file, bed_file)
var_dl = DataLoader(var_ds, shuffle=False, batch_size=bs, num_workers=1, pin_memory=True)#, #collate_fn=geneslice_collate_fn)

# Run
preds = []
with torch.inference_mode():
    with torch.autocast(device, enabled=not disable_autocast):
        for batch in tqdm.tqdm(var_dl, miniters=10):
            seq, bins = batch
            seq = seq.to(device)
            seq = seq.permute(0,2,1)
            # predict sense and antisense and average
            bins = bins.squeeze()
            try:
                if bins.shape[0] == 1:
                    print (bins, "zero gene slice")
                    # continue
            except:
                print (bins)
                preds.append(counts_outputs_rna)
                continue
            # bins = bins.tolist()
            # print (bins)
            counts_outputs_rna = get_pseudobulk_count_pred(scooby, seq, gene_slice=bins, 
                                                     strand='+', predict=predict, clip_soft=5, 
                                                     model_type = "rna",
                                                     cell_emb_conv_weights_and_biases=cell_emb_conv_weights_and_biases).cpu()
            #print(counts_outputs_rna.shape)
            counts_outputs_rna = torch.log(counts_outputs_rna+1)
            preds.append(counts_outputs_rna)

preds = torch.stack(preds)
snp_effects = (preds[len(var_ds.snp_df):] - preds[:len(var_ds.snp_df)])

adata = ad.AnnData(snp_effects.numpy(), 
                   obs=var_ds.snp_df.copy(),
                   var=cell_type_index[['celltype']].set_index('celltype'),
                  )
adata.write(out_path, compression="gzip")
