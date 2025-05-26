import time
import pandas as pd
import os
import numpy as np
import snapatac2_scooby as sp
import scanpy as sc
import anndata as ad
import glob

import scipy.sparse
import tqdm

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--number", type=int, help="Sample Number argument")
args = parser.parse_args()


## Read in RNA bam files

data_path = 'onek1k_bam_files/snapatac/fragments/'

adata = sc.read_h5ad(
    os.path.join('/data/ceph/hdd/project/node_08/QNA/scborzoi/submission_data', 'onek1k_training_data', 'OneK1K_only_immune.h5ad'), backed='r')

### Make Fragment file

sample_files = glob.glob(os.path.join(data_path , 'processed', 'filtered_bam/*bam'))

sample_files =  [sample_files[args.number]]

for sample_file in tqdm.tqdm(sample_files):
    print(sample_file)
    sample = os.path.basename(sample_file).split('_tag.bam')[0] # now, we can just use the xf_filter flag in sp.pp.make_fragment_file
    print(sample)
    out_path = os.path.join(data_path, 'snapatac', 'fragments')
    outfile = os.path.join(out_path, f'{sample}.fragments.bed.gz')
    whitelist = adata.obs.query("sample_name == @sample")['barcode'].to_list()
    sample_number = adata.obs.query("sample_name == @sample")['sample'][0]
    
    print(sample_number)
    if (len(whitelist)>0) & (~os.path.exists(os.path.join(out_path, f'{sample}.fragments.bed.minus.gz'))):
        sp.pp.make_fragment_file(
            sample_file, 
            output_file=outfile,
            barcode_tag="CB", 
            umi_tag="UB",
            umi_regex=None, 
            stranded=True, 
            is_paired=False, 
            shift_left=0, 
            shift_right=0
        )
        

    for strand in ['plus', 'minus']:
        if not os.path.exists(os.path.join(data_path, 'snapatac', 'anndata', f'snapatac_{sample}_{strand}.h5ad')):
            test = sp.pp.import_data(
                    f"{out_path}/{sample}.fragments.bed.{strand}.gz", 
                    chrom_sizes=sp.genome.hg38, 
                    min_num_fragments=0, 
                    n_jobs=-1,
                    whitelist=whitelist
                )
            if not test.obs_names.isin(whitelist).all():
                print(sample)
                break
            test.obs.index = test.obs.index + '_' + sample_number
                
            test.X = scipy.sparse.csr_matrix((test.obsm['fragment_single'].shape[0], 0))
            test.write(os.path.join(data_path, 'snapatac', 'anndata', f'snapatac_{sample}_{strand}.h5ad'))
    else:
        print(f"Skipping {sample}")