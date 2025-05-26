import os

import numpy as np
import pandas as pd
import scipy
import torch
import tqdm
from accelerate import Accelerator
import polars as pl
import scanpy as sc

from torch.utils.data import DataLoader

from enformer_pytorch.data import GenomeIntervalDataset

from scooby.modeling import Scooby
from scooby.data import onTheFlyMultiomeDataset
from scooby.utils.utils import fix_rev_comp_multiome, undo_squashed_scale, get_gene_slice_and_strand
from scooby.utils.transcriptome import Transcriptome

from tangermeme.io import read_meme
from tangermeme.tools.fimo import fimo
from tangermeme.ersatz import randomize, substitute
from tangermeme.utils import characters


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

def find_and_replace_motif(seqs_ref, motif, substitution):    # compute motif hits
    hits = fimo(motif, seqs_ref)[0]
    seqs_alt = seqs_ref.clone()
    if len(hits) > 0:
        for _, motif_range in hits.iterrows():
            seqs_alt = substitute(seqs_alt, substitution, start=motif_range.start, allow_N=True)
            
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

    accelerator = Accelerator(step_scheduler_with_optimizer=False)
    
    import yaml
    with open("../../training_epicardioids/config_multiome.yaml", "r") as f:
            config = yaml.safe_load(f)
    
    # Extract configuration parameters
    data_path = "/data/ceph/hdd/project/node_08/QNA/scborzoi/submission_data"
    local_world_size = 1
    embedding_path = config["data"]["embedding_path"]
    neighbors_path = config["data"]["neighbors_path"]
    sequences_path = config["data"]["sequences_path"]
    genome_path = config["data"]["genome_path"]
    
    cell_emb_dim = config["model"]["cell_emb_dim"]
    num_tracks = config["model"]["num_tracks"]
    context_length = config["data"]["context_length"]

    csb = Scooby.from_pretrained(
    os.path.join(data_path, 'epicardioids-scooby'),
    cell_emb_dim=cell_emb_dim,
    embedding_dim=1920,
    n_tracks=num_tracks,
    return_center_bins_only=True,
    disable_cache=False,
    use_transform_borzoi_emb=True,
)
    csb = accelerator.prepare(csb)
    csb.eval()
    
    gtf_file = os.path.join(data_path, "gencode.v32.annotation.sorted.gtf.gz")
    transcriptome = Transcriptome(gtf_file, use_geneid=True)

    neighbors = scipy.sparse.load_npz(neighbors_path)
    embedding = pd.read_parquet(embedding_path)
    
    cell_type_index = pd.read_parquet(os.path.join(data_path, "epicardioids_training_data", 'celltype.pq'))
    
    cell_type_index['celltype'] = cell_type_index['celltype'].str.replace(' ', '_').str.replace("/", "_")
    cell_type_index = cell_type_index.sort_values('celltype')

    clip_soft=5
    adata = sc.read(os.path.join(data_path, 'epicardioids_training_data', 'adata_matched.h5ad'), gex_only=False, backed='r')

    embeddings = torch.from_numpy(
            np.vstack(
                embedding['embedding'].values # gets embeddings of all cells of the cell type
                )
            ).cuda()
    filter_val = lambda df: df.filter(pl.col('column_2') >=0)
    val_ds = GenomeIntervalDataset(
        bed_file = os.path.join(data_path, "epicardioids_training_data", 'DEG_gene_sequences.csv'),
        fasta_file = genome_path,
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

    counts_ref_rna, counts_alt_rna = [], []
    counts_ref_atac, counts_alt_atac = [], []
    
    # iterate over all val gene sequences
    for i,x in tqdm.tqdm(enumerate(val_loader), disable = False, total=len(val_dataset)): 
        gene_slice, strand = get_gene_slice_and_strand(transcriptome, val_dataset.genome_ds.df[i, 'column_4'], val_dataset.genome_ds.df[i, 'column_2'], span = True)
        
        seqs = x[0].cuda().permute(0,2,1)
        seqs_rev_comp =  torch.flip(seqs.permute(0,2,1), (-1, -2)).permute(0,2,1)
    
        
        stacked_counts_ref_rna, stacked_counts_alt_rna = [], []
        stacked_counts_ref_atac, stacked_counts_alt_atac = [], []
       
        for conv_weight, conv_bias in cell_emb_conv_weights_and_biases:
            counts_substition_rna, counts_substition_atac = [], []
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
    ad_alt_motif = ad_alt[adata.obs_names].copy()
    ad_alt_motif.write(os.path.join(data_path, 'motif_effects/epicardiods_motif_effects/ref_multiome.h5ad'))
