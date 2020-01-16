#!/bin/bash

#$ -N anvil-train
#$ -cwd
#$ -l h_vmem=8G
#$ -l h_rt=12:00:00
#$ -pe sharedmem 4

source /etc/profile.d/modules.sh
module load igmm/apps/miniconda3/4.5.11

source /exports/igmm/software/pkg/el7/apps/miniconda3/4.5.11/etc/profile.d/conda.sh
conda activate anvil-dev

bash ./train_to_acceptance.sh ./runcards/train.yml ./runcards/short_sample.yml

