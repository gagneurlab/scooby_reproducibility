#!/bin/bash -l
chunk=$1

source ~/.bashrc
conda activate scooby_reproducibility

python -u parallel_inference.py $chunk
