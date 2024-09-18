#!/bin/bash

source ~/.bashrc
for i in {0..3}
do
   sbatch --job-name=Motif_chunk_predictor -x ouga08,ouga05 --nodes=1 --partition standard --cpus-per-task=8 --mem=48GB --requeue  --gres=gpu:1 inference_bash.sh $i &
done
