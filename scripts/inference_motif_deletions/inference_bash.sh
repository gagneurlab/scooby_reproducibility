#!/bin/bash -l
chunk=$1

source ~/.bashrc
conda activate borzoi-pytorch

python -u parallel_inference-rnaonly.py $chunk
