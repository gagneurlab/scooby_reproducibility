# Scooby Reproducibility
Code for scooby manuscript. Scooby is the first model to predict scRNA-seq coverage and scATAC-seq insertion profiles along the genome at single-cell resolution. For this, it leverages the pre-trained multi-omics profile predictor Borzoi as a foundation model, equips it with a cell-specific decoder, and fine-tune its sequence embeddings. Specifically, the decoder is conditioned on the cell position in a precomputed single-cell embedding.

This repository contains notebooks and example code to reproduce the results of the [manuscript](https://www.biorxiv.org/content/10.1101/2024.09.19.613754v1). The [model repository](https://github.com/gagneurlab/scooby/tree/main) contains model and data loading code and a train script.

Hardware requirements
---------------------

-  NVIDIA GPU (tested on A40), Linux, Python (tested with v3.9)


Installation instructions
-------------------------

- ``pip install -r requirements.txt``
-  Download file contents from the Zenodo
   [repo](https://zenodo.org/records/14051793)
