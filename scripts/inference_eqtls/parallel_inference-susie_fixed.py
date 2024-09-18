import os

import numpy as np
import pandas as pd
import polars as pl
import scipy

import torch
import tqdm
from accelerate import Accelerator
from polya_project.data import GenomeIntervalDataset, str_to_one_hot
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

from utils.utils import fix_rev_comp_multiome, get_gene_slice_and_strand, get_cell_count_pred
from modeling.scborzoi import ScBorzoi
from data.scdata import onTheFlyExonMultiomePseudobulkDataset

def get_pred_for_snp(row):
    try:
        gene = row['gene_name']
        hg38_pos = row['Start']
        alt = row['nuc']
        genes = [gene]
    
        filter_val = lambda df: df.filter(pl.col('column_4').is_in(genes))#
        #filter_val = lambda df: df.filter(True)
        val_ds = GenomeIntervalDataset(
            bed_file = os.path.join(data_path,'borzoi_training_data_fixed', 'train_val_test_gene_sequences.csv'),
            fasta_file = fasta_file,
            filter_df_fn = filter_val,
            return_seq_indices = False,
            shift_augs = (0,0),
            rc_aug = False,
            return_augs = True,
            context_length = context_length,
            chr_bed_to_fasta_map = {}
        )
        val_dataset = onTheFlyExonMultiomePseudobulkDataset(
            cell_types = cell_type_index['celltype'],
            ds = val_ds, 
            base_path = base_path,
            seqlevelstyle="UCSC",
            clip_soft=clip_soft
        )
        
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle = False, num_workers = 0)
        val_loader = accelerator.prepare(val_loader)
    
    
        alt = str_to_one_hot(alt).cuda() 
        
        start = val_ds.df['column_2'].item() - ((context_length - (6144*32))//2)
        
        snp_idx = hg38_pos - start - 1
        if snp_idx > 524287 or snp_idx < 0:
            print(gene)
            return None
        
        counts_outputs_rna = []
        
        # iterate over all val gene sequences
        for i,x in tqdm.tqdm(enumerate(val_loader), disable = True, total=len(val_dataset)):   
            gene_slice, strand = get_gene_slice_and_strand(
                transcriptome,  
                val_dataset.genome_ds.df[i, 'column_4'],     
                val_dataset.genome_ds.df[i, 'column_2'], 
                span=True
            )
            
            bs = x[0].shape[0]
            seqs = x[0].cuda().permute(0,2,1)
        
            seqs_alt = seqs.clone()
            if row['type'] == 'ref':
                print (seqs_alt[:, :, snp_idx], row['nuc'])
            seqs_alt[:, :, snp_idx] = alt 
    
            counts_outputs_rna = get_cell_count_pred(csb, seqs_alt, gene_slice=gene_slice, strand=strand, predict=predict,clip_soft=clip_soft, model_type = "multiome_rna",region_slice = None, conv_weight=conv_weights, conv_bias=conv_biases, num_neighbors=1, chunk_size = 70000)['rna'].cpu()
            return counts_outputs_rna.clone().numpy(force=True)
    except Exception as e: 
        print(e, 'ERROR')
        return None


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
    outputs_rev_comp = fix_rev_comp_multiome(flipped_version) #fix_rev_comp2(flipped_version)
    #outputs_rev_comp = fix_rev_comp2(flipped_version) #fix_rev_comp2(flipped_version)
    return (outputs + outputs_rev_comp)/2


if __name__ == '__main__' :
    accelerator = Accelerator(step_scheduler_with_optimizer=False)

    data_path = 'tmp'
    csb = ScBorzoi(
        cell_emb_dim=14, 
        n_tracks=3,
        embedding_dim=1920, 
        return_center_bins_only=True, 
        disable_cache=True,
        use_transform_borzoi_emb=True
    )
    old_weights = torch.load('/s/project/QNA/borzoi/f0/model0_best.h5.pt')
    csb.load_state_dict(old_weights, strict = False)
    
    config = LoraConfig(
        target_modules=r"(?!separable\d+).*conv_layer|.*to_q|.*to_v|transformer\.\d+\.1\.fn\.1|transformer\.\d+\.1\.fn\.4",
    )
    csb = get_peft_model(csb, config)
    csb.load_state_dict(torch.load(os.path.join(data_path, 'borzoi_saved_models/csb_epoch_20_scDog-neurips-PMseq-4nodrop-softclip5-64cell-normalizeATAC-fixedemb-noneighbors-rightembeddingrightsplit-longer/pytorch_model.bin'))) 
    csb = csb.merge_and_unload()
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

    embeddings = torch.from_numpy(
            np.vstack(
                embedding['embedding'].values # gets embeddings of all cells of the cell type
                )
            ).cuda()
    conv_weights, conv_biases = csb.forward_cell_embs_only(embeddings.unsqueeze(0))
    snp_df = pd.read_csv(os.path.join(data_path, "annotations/susie_snps_fixed.csv"))
    snp_df['pred'] = snp_df.apply(lambda x : get_pred_for_snp(x), axis = 1)
    
    snp_df.to_pickle(os.path.join("annotations/susie_snps_preds_fixed.pickle"))
