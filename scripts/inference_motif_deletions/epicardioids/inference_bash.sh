#!/bin/bash 
chunk=$1

CONDA_ACTIVATE_PATH=$(conda info --base)/etc/profile.d/conda.sh
source $CONDA_ACTIVATE_PATH
conda activate scooby_package

python -u parallel_inference.py $chunk
