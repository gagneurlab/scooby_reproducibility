# Scooby Reproducibility
Code for scooby manuscript. Scooby is the first model to predict scRNA-seq coverage and scATAC-seq insertion profiles along the genome at single-cell resolution. For this, it leverages the pre-trained multi-omics profile predictor Borzoi as a foundation model, equips it with a cell-specific decoder, and fine-tune its sequence embeddings. Specifically, the decoder is conditioned on the cell position in a precomputed single-cell embedding.

This repository contains notebooks and example code to reproduce the results of the manuscript. The [model repository](https://github.com/gagneurlab/scooby/tree/main) contains model and data loading code and a train script.
