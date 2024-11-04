"""
this will write out profiles for a cell with its surrounding 100 nearest neighbors of an indexed cell of cells_for_profile_eval.csv.
Note that the resulting profiles will not be squashed scaled.

Adapt the paths below and call it like 'python write_out_profile_targets_for1D.py 0' to get the profiles for the 0th cell in cells_for_profile_eval.csv etc.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import torch
import tqdm

from torch.utils.data import DataLoader

from enformer_pytorch.data import GenomeIntervalDataset,FastaInterval
from scooby.utils.transcriptome import Transcriptome

from typing import Optional
import polars as pl
import scipy.sparse
from torch.utils.data import Dataset

min_value = torch.finfo(torch.float16).min
max_value = torch.finfo(torch.float16).max

min_value = torch.finfo(torch.float16).min
max_value = torch.finfo(torch.float16).max


ONE_BY_NINETY = (1 / 90)
def _sparse_to_coverage_rna(m, seq_coord, strand):
    _, _, chrom_end, start, end, seq_coord_2, seq_coord_3 = seq_coord
    m = m[:, start:end]
    # Initialize dense matrix with zeros
    dense_matrix = np.zeros(m.shape, dtype=np.single)
    # Iterate over non-zero elements of the sparse matrix
    if strand == "plus":
        for row in range(m.shape[0]):
            col_indices = m.indices[m.indptr[row] : m.indptr[row + 1]]
            values = m.data[m.indptr[row] : m.indptr[row + 1]]
            for col_index, value in zip(col_indices, values):
                dense_matrix[row, col_index : (col_index + value)] += ONE_BY_NINETY
    elif strand == "minus":
        for row in range(m.shape[0]):
            col_indices = m.indices[m.indptr[row] : m.indptr[row + 1]]
            values = m.data[m.indptr[row] : m.indptr[row + 1]]
            for col_index, value in zip(col_indices, values):
                dense_matrix[row, (col_index + value + 1) : (col_index + 1)] += ONE_BY_NINETY
    # restrict to relevant part
    dense_matrix = dense_matrix[:, min([100, seq_coord_2]) : max([-100, seq_coord_3 - chrom_end])]
    dense_matrix = torch.from_numpy(dense_matrix)#.unsqueeze(0)
    return dense_matrix


def _sparse_to_coverage_atac(m, seq_coord):
    _, _, chrom_end, start, end, seq_coord_2, seq_coord_3 = seq_coord
    m = m[:, start:end]
    dense_matrix = m.sum(0).astype(np.single).A[0]
    # restrict to relevant part
    dense_matrix = dense_matrix[min([100, seq_coord_2]) : max([-100, seq_coord_3 - chrom_end])]
    # For ATAC it is easy because we can just use the matrix as is
    dense_matrix = torch.from_numpy(dense_matrix).unsqueeze(0)

    return dense_matrix



class onTheFlyMultiomeDataset(Dataset):  # noqa: D101
    def __init__(
        self,
        adatas: dict,
        neighbors: scipy.sparse.csr_matrix,
        embedding: pd.DataFrame,
        ds: GenomeIntervalDataset,
        clip_soft,
        cell_sample_size: int = 32,
        get_targets: bool = True,
        random_cells: bool = True,
        cells_to_run: Optional[np.ndarray] = None,
        cell_weights: Optional[np.ndarray] = None,
        normalize_atac: bool = False,
    ) -> None:
        self.clip_soft = clip_soft
        self.neighbors = neighbors
        self.cell_weights = cell_weights
        self.cells_to_run = cells_to_run
        self.embedding = embedding
        self.get_targets = get_targets
        self.random_cells = random_cells
        if not self.random_cells and not cells_to_run:
            self.cells_to_run = np.zeros(1, dtype=np.int64)
        self.genome_ds = ds
        self.cell_sample_size = cell_sample_size
        self.adatas = adatas
        self.normalize_atac = normalize_atac
        self.neighbor_cache = dict()

        try:
            self.chrom_sizes = self.adatas["rna_plus"].uns["reference_sequences"].copy()
            if "chr" not in self.chrom_sizes["reference_seq_name"][0]:
                # convert to chr1, chr2, etc
                self.chrom_sizes["reference_seq_name"] = "chr" + self.chrom_sizes["reference_seq_name"].astype(str)
            self.chrom_sizes["offset"] = np.insert(self.chrom_sizes["reference_seq_length"].cumsum()[:-1].values, 0, 0)
            self.chrom_sizes = self.chrom_sizes.set_index("reference_seq_name").to_dict("index")
        except:
            pass

    def __len__(self):
        return len(self.genome_ds)

    def _get_neighbors_for_cell(self, bar_code_id):  # noqa: D102
        if bar_code_id not in self.neighbor_cache.keys():
            cell_neighbor_ids = self.neighbors[bar_code_id].nonzero()[1].tolist() + [bar_code_id]
            self.neighbor_cache[bar_code_id] = cell_neighbor_ids
        else:
            cell_neighbor_ids = self.neighbor_cache[bar_code_id]
        neighbors_to_load = cell_neighbor_ids
        return neighbors_to_load

    def _process_rna(self, adata, cell_indices, seq_coord, strand):
        tensor = _sparse_to_coverage_rna(
            m=adata.obsm["fragment_single"][cell_indices], seq_coord=seq_coord, strand=strand
        )
        seq_cov = torch.nn.functional.avg_pool1d(tensor, kernel_size=32, stride=32) * 32
        # seq_cov = -1 + (1 + seq_cov) ** 0.75

        # clip_soft = self.clip_soft
        # clip = 768

        # clip_mask = seq_cov > clip_soft
        # if clip_mask.any():
        #     seq_cov[clip_mask] = clip_soft - 1 + torch.sqrt(seq_cov[clip_mask] - clip_soft + 1)
        # seq_cov = torch.clip(seq_cov, -clip, clip)
        return seq_cov

    def _process_atac(self, adata, cell_indices, seq_coord):
        tensor = _sparse_to_coverage_atac(m=adata.obsm["insertion"][cell_indices], seq_coord=seq_coord)
        seq_cov = torch.nn.functional.avg_pool1d(tensor, kernel_size=32, stride=32) * 32
        return seq_cov

    def _load_pseudobulk(self, neighbors, seq_coord):
        # process all modalities
        seq_covs = []
        for modality, adata in self.adatas.items():
            if "rna" in modality:
                strand = modality.split("_")[-1]
                seq_cov = self._process_rna(adata, neighbors, seq_coord, strand=strand)
            elif "atac" in modality:
                seq_cov = self._process_atac(adata, neighbors, seq_coord)
            seq_cov = seq_cov.mean(0, keepdims= True)
            seq_covs.append(seq_cov)
        return torch.cat(seq_covs)

    def _reinit_fasta_reader(self):
        self.genome_ds.fasta = FastaInterval(
            fasta_file=self.genome_ds.fasta.seqs.filename,
            context_length=self.genome_ds.fasta.context_length,
            return_seq_indices=self.genome_ds.fasta.return_seq_indices,
            shift_augs=self.genome_ds.fasta.shift_augs,
            rc_aug=self.genome_ds.fasta.rc_aug,
        )

    def __getitem__(self, idx):
        self._reinit_fasta_reader()
        if self.random_cells:
            idx_cells = np.random.choice(self.neighbors.shape[0], size=self.cell_sample_size, p=self.cell_weights)
        else:
            idx_cells = self.cells_to_run
        idx_gene = idx
        seq_coord = self.genome_ds.df[idx_gene]
        inputs, _, rc_augs = self.genome_ds[idx_gene]
        embeddings = torch.from_numpy(np.vstack(self.embedding.iloc[idx_cells]["embedding"].values))

        if self.get_targets:
            chrom_size = self.chrom_sizes[seq_coord["column_1"].item()]
            chrom_start = chrom_size["offset"]
            chrom_end = chrom_size["reference_seq_length"]

            seq_coord_2, seq_coord_3 = seq_coord["column_2"].item(), seq_coord["column_3"].item()
            start = np.max([0, seq_coord_2 - 100]) + chrom_start
            end = np.min([seq_coord_3 + 100, chrom_end]) + chrom_start
            genome_data = [chrom_size, chrom_start, chrom_end, start, end, seq_coord_2, seq_coord_3]

            targets = []
            for cell_idx in tqdm.tqdm(idx_cells, disable=True):
                neighbors_to_load = self._get_neighbors_for_cell(cell_idx)
                targets.append(self._load_pseudobulk(neighbors_to_load, genome_data))
            targets = torch.vstack(targets)
            return inputs, rc_augs, targets.permute(1, 0), embeddings
        return inputs, rc_augs, embeddings


if __name__ == '__main__' :
    import argparse

    # Instantiate the parser
    parser = argparse.ArgumentParser(description='no')
    # Required positional argument
    parser.add_argument('cell_idx', type=int,
                        help='cell_index')
    args = parser.parse_args()
    
    max_cell = args.cell_idx
    print (max_cell)

    data_path = '/s/project/QNA/scborzoi/neurips_bone_marrow'

    
    transcriptome = Transcriptome(f'{data_path}/gencode.v32.annotation.gtf')
    adata_plus = sc.read(f'{data_path}/scooby_training_data/snapatac_merged_fixed_plus.h5ad')
    adata_minus = sc.read(f'{data_path}/scooby_training_data/snapatac_merged_fixed_minus.h5ad')
    adata_atac = sc.read(f'{data_path}/scooby_training_data/snapatac_merged_fixed_atac.h5ad')
    cell_type_index = pd.read_parquet(os.path.join(data_path,  'scooby_training_data/celltype_fixed.pq'))
    cell_type_index['size'] = cell_type_index['cellindex'].apply(lambda x: len(x))
    embedding = pd.read_parquet(f'{data_path}/scooby_training_data/embedding_no_val_genes_new.pq')
    fasta_file = os.path.join(data_path,'genome_human.fa')
    bed_file = os.path.join(data_path,"sequences.bed")
    neighbors = scipy.sparse.load_npz('neighbors_100_no_val_genes_new.npz')
    adatas = {'rna_plus' : adata_plus, 'rna_minus' : adata_minus, 'atac': adata_atac,}
    
    filter_val = lambda df: df.filter((pl.col('column_4') == 'fold3') )
    val_ds = GenomeIntervalDataset(
            bed_file = bed_file,
            fasta_file = fasta_file,                        
            filter_df_fn = filter_val,                       
            return_seq_indices = False,                         
            shift_augs = (0,0),                              
            rc_aug = False,
            return_augs = True,
            context_length = 524288,
            chr_bed_to_fasta_map = {}
        )
    len(val_ds)

    cells_to_run = [pd.read_csv('cells_for_profile_eval.csv')['0'].values[max_cell]]
    
    val_dataset = onTheFlyMultiomeDataset(
        adatas=adatas,
        neighbors=neighbors, 
        embedding=embedding, 
        ds=val_ds, 
        get_targets= True, 
        random_cells = False,  
        cells_to_run = sorted(list(cells_to_run)),
        cell_sample_size = 1,
        normalize_atac = True,
        clip_soft = 5
    )
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle = False, num_workers = 0)

    all_cells_all_genes = []
    for i in tqdm.tqdm(range(len(val_ds.df))):
        all_cells_all_genes.append(val_dataset[i][-2])
    torch.save(torch.vstack(all_cells_all_genes), f"profiles_{args.cell_idx}_target.pt") 
