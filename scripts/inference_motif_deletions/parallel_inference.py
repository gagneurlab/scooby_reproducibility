import numpy as np
import pandas as pd
import polars as pl
import scipy
import scanpy as sc
import os

import torch
import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader

from scooby.utils.utils import fix_rev_comp_multiome, undo_squashed_scale
from scooby.modeling import Scooby
from scooby.data import onTheFlyMultiomeDataset
from enformer_pytorch.data import GenomeIntervalDataset

from tangermeme.io import read_meme
from tangermeme.tools.fimo import FIMO
from tangermeme.ersatz import randomize, substitute
from tangermeme.utils import characters

def compute_gene_expression(
    gene_slice,
    strand,
    csb, 
    seqs, 
    seqs_rev_comp, 
    conv_weight, 
    conv_bias 
):
    outputs = predict(csb, seqs, seqs_rev_comp, conv_weight, conv_bias, bins_to_predict= gene_slice)
    # get RNA:
    outputs_rna = outputs[:,:,torch.tensor([1,1,0]).repeat(outputs.shape[2]//3).bool()]
    outputs_rna = outputs_rna.float().detach()
    
    # sum exons on positive/negative strand for all cells
    num_pos = outputs.shape[-1]
    if strand == '+':
        unsquashed = undo_squashed_scale(outputs_rna[0, : ,:num_pos:2], clip_soft=5)
    elif strand == '-':
        unsquashed = undo_squashed_scale(outputs_rna[0, : ,1:num_pos:2], clip_soft=5)
    return unsquashed.sum(axis=0).detach().clone().cpu()

def compute_rna_atac(
    gene_slice,
    strand,
    csb, 
    seqs, 
    seqs_rev_comp, 
    conv_weight, 
    conv_bias 
):
    outputs = predict(csb, seqs, seqs_rev_comp, conv_weight, conv_bias)
    # get RNA:
    outputs_rna = outputs[:,:,torch.tensor([1,1,0]).repeat(outputs.shape[2]//3).bool()]
    outputs_rna = outputs_rna.float().detach()[:,gene_slice,:]
    
    # sum exons on positive/negative strand for all cells
    num_pos = outputs.shape[-1]
    if strand == '+':
        unsquashed = undo_squashed_scale(outputs_rna[0, : ,:num_pos:2], clip_soft=5)
    elif strand == '-':
        unsquashed = undo_squashed_scale(outputs_rna[0, : ,1:num_pos:2], clip_soft=5)
        
    # get ATAC:
    outputs_atac = outputs[:,:,torch.tensor([0,0,1]).repeat(outputs.shape[2]//3).bool()]
    outputs_atac = outputs_atac.float().detach() * 20
    return unsquashed.sum(axis=0).detach().clone().cpu(), outputs_atac[0].sum(axis=0).detach().clone().cpu()


def compute_accessibility(
    csb, 
    seqs, 
    seqs_rev_comp, 
    conv_weight, 
    conv_bias 
):
    outputs = predict(csb, seqs, seqs_rev_comp, conv_weight, conv_bias)
    # get ATAC:
    outputs_atac = outputs[:,:,torch.tensor([0,0,1]).repeat(outputs.shape[2]//3).bool()]
    outputs_atac = outputs_atac.float().detach() * 20
    return outputs_atac[0].sum(axis=0).detach().clone().cpu()

def find_and_replace_motif(seqs_ref, model, substitution):    # compute motif hits
    hits = model.hits(seqs_ref)[0]
    seqs_alt = seqs_ref.clone()
    if len(hits) > 0:
        for _, motif_range in hits.iterrows():
            seqs_alt = substitute(seqs_alt, substitution, start=motif_range.start)
            
    seqs_alt_rev_comp =  torch.flip(seqs_alt.permute(0,2,1), (-1, -2)).permute(0,2,1)
    return seqs_alt, seqs_alt_rev_comp 

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
    outputs_rev_comp = fix_rev_comp_multiome(flipped_version)
    return (outputs + outputs_rev_comp)/2


if __name__ == '__main__' :
    import argparse

    # Instantiate the parser
    parser = argparse.ArgumentParser(description='no')
    # Required positional argument
    parser.add_argument('chrom', type=int,
                        help='A required integer positional argument')
    args = parser.parse_args()
    motif_to_do = pd.read_csv("MotifTable.csv", header = None).iloc[args.chrom].item()
    accelerator = Accelerator(step_scheduler_with_optimizer=False)
    n_sub = 10
    subset = 10
    
    data_path = 'tmp'
    csb = Scooby.from_pretrained('johahi/neurips-scooby', cell_emb_dim = 14, n_tracks = 3, use_transform_borzoi_emb = True)
    csb = accelerator.prepare(csb)
    csb.eval()
    gtf_file = os.path.join(data_path, "gencode.v32.annotation.sorted.gtf.gz")
    fasta_file = os.path.join(data_path, 'genome_human.fa')
    bed_file = os.path.join(data_path, 'sequences.bed')
    import pickle
    with open(os.path.join(data_path, 'gencode.v32.annotation.gtf.transcriptome'), 'rb') as handle:
        transcriptome = pickle.load(handle)
    base_path = os.path.join(data_path, 'snapatac', 'pseudobulks/')
    sample = 'merged'
    neighbors_100 = scipy.sparse.load_npz(os.path.join(data_path, 'borzoi_training_data_fixed', 'neighbors_100_no_val_genes_new.npz'))
    embedding = pd.read_parquet(os.path.join(data_path, 'borzoi_training_data_fixed',  'embedding_no_val_genes_new.pq'))
    neighbors = scipy.sparse.csr_matrix(neighbors_100.shape)
    cell_type_index = pd.read_parquet(os.path.join(data_path,  'borzoi_training_data_fixed/celltype_fixed.pq'))
    cell_type_index['size'] = cell_type_index['cellindex'].apply(lambda x: len(x))
    cell_type_index['celltype_name'] = cell_type_index['celltype'].copy()
    cell_type_index['celltype'] = cell_type_index['celltype'].str.replace(' ', '_').replace(r"G/M_prog", "G+M_prog").replace("MK/E_prog", "MK+E_prog")
    cell_type_index = cell_type_index.sort_values('celltype')
    cell_type_index = cell_type_index.reset_index(drop=True)
    context_length = 524288 
    clip_soft=5

    adata = sc.read(os.path.join(data_path, 'bmmc_multiome_multivi_neurips21_curated_new_palantir_fixed.h5ad'))

    embeddings = torch.from_numpy(
            np.vstack(
                embedding['embedding'].values # gets embeddings of all cells of the cell type
                )
            ).cuda()
    filter_val = lambda df: df.filter(pl.col('column_2') >=0)
    val_ds = GenomeIntervalDataset(
        bed_file = os.path.join(data_path, 'borzoi_training_data_fixed/DEG_gene_sequences.csv'),
        fasta_file = fasta_file,
        filter_df_fn = filter_val,
        return_seq_indices = False,
        shift_augs = (0,0),
        rc_aug = False,
        return_augs = True,
        context_length = context_length,
        chr_bed_to_fasta_map = {}
    )
    val_dataset = onTheFlyMultiomeDataset(
        adatas=None,
        neighbors=neighbors,
        embedding=embedding,
        ds = val_ds, 
        clip_soft=5,
        get_targets=False
    )
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle = False, num_workers = 0)
    csb, val_loader = accelerator.prepare(csb, val_loader)
    
    csb.eval()
    cell_indices  = []
    for _, row in tqdm.tqdm(cell_type_index.iterrows(),disable = True):
        cell_indices.append(
            torch.from_numpy(
                np.vstack(
                    embedding.iloc[row['cellindex']]['embedding'].values # gets embeddings of all cells of the cell type
                    )
                ).unsqueeze(0)
            ) # prep cell_embeddings
    
    # get conv weights and biases for all cells sorted by cell type in a list
    cell_emb_conv_weights_and_biases =[]
    for cell_emb_idx in tqdm.tqdm(cell_indices, disable = True):
        with torch.no_grad():
            cell_emb_idx = cell_emb_idx.cuda()
            conv_weights, conv_biases = csb.forward_cell_embs_only(cell_emb_idx)
            cell_emb_conv_weights_and_biases.append((conv_weights, conv_biases))
    pwms = read_meme(
    os.path.join(data_path, "annotations/motif_databases/HOCOMOCO/H12CORE_meme_format.meme")

    )
    motif_name = motif_to_do
    motif = {motif_name: pwms[motif_name]}
    motif_oh = torch.from_numpy(
        (motif[motif_name] == motif[motif_name].max(axis=1)[:, np.newaxis]).astype(int)
    ).T.unsqueeze(0)
    print(characters(motif_oh[0]))
    
    oh_random_substitutions = []
    for seed in range(n_sub):
        oh_random_substitutions.append(randomize(
            motif_oh, 
            start=0, 
            end=(motif_oh.shape[-1] - 1),
            random_state=seed
        )[0])
    model = FIMO(motif)
    counts_ref_rna, counts_alt_rna = [], []
    counts_ref_atac, counts_alt_atac = [], []
    
    # iterate over all val gene sequences
    for i,x in tqdm.tqdm(enumerate(val_loader), disable = False, total=len(val_dataset)):   
        #if i == subset:
        #    break
        gene_slice = transcriptome.genes[
        val_dataset.genome_ds.df[i, 'column_4']
        ].output_slice(
            val_dataset.genome_ds.df[i, 'column_2'],
            6144*32,
            32,
            span=True
            ) #select right columns
        strand = transcriptome.genes[val_dataset.genome_ds.df[i, 'column_4']].strand
        
        bs = x[0].shape[0]
        seqs = x[0].cuda().permute(0,2,1)
        seqs_rev_comp =  torch.flip(seqs.permute(0,2,1), (-1, -2)).permute(0,2,1)
    
        
        stacked_counts_ref_rna, stacked_counts_alt_rna = [], []
        stacked_counts_ref_atac, stacked_counts_alt_atac = [], []
    
        alt_seqs = [find_and_replace_motif(seqs, model, sub) for sub in oh_random_substitutions]
       
        for conv_weight, conv_bias in cell_emb_conv_weights_and_biases:
            counts_substition_rna, counts_substition_atac = [], []
            for seqs_alt, seqs_alt_rev_comp in alt_seqs:
                rna, atac = compute_rna_atac(
                    gene_slice,
                    strand,
                    csb, 
                    seqs, 
                    seqs_rev_comp, 
                    conv_weight, 
                    conv_bias 
                )
                counts_substition_rna.append(rna)
                counts_substition_atac.append(atac)
                break
            stacked_counts_alt_rna.append(torch.stack(counts_substition_rna).mean(axis=0)) # RNA counts
            stacked_counts_alt_atac.append(torch.stack(counts_substition_atac).mean(axis=0)) # ATAC counts
        
        counts_alt_rna.append(torch.concat(stacked_counts_alt_rna).detach().clone().cpu())
        counts_alt_atac.append(torch.concat(stacked_counts_alt_atac).detach().clone().cpu())
    counts_alt_rna = torch.stack(counts_alt_rna, axis=1).clone().numpy(force=True)
    counts_alt_atac = torch.stack(counts_alt_atac, axis=1).clone().numpy(force=True)
    var = val_dataset.genome_ds.df.to_pandas()
    var = var.set_index('column_4')
    ## get obs names
    cell_indices = np.concatenate(cell_type_index['cellindex'].values)
    obs = adata[cell_indices].obs
    ad_alt = sc.AnnData(counts_alt_rna, obs=obs, var=var)
    ad_alt.layers['atac'] = counts_alt_atac
    ad_alt_gata = ad_alt[adata.obs_names].copy()
    ad_alt_gata.write(os.path.join(data_path, 'motif_deletion/alt_{motif_name}_multiome_fixed.h5ad'))
