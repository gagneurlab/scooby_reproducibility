#!/bin/bash

# CONDA_ACTIVATE_PATH=$(conda info --base)/etc/profile.d/conda.sh
# source $CONDA_ACTIVATE_PATH
# conda activate scooby_reproducibility
for i in {0..1}
do
   sbatch --job-name=Motif_chunk_predictor -x ouga08,ouga05 --nodes=1 --partition standard --cpus-per-task=8 --mem=48GB --gres=gpu:1 inference_bash.sh $i &
done
