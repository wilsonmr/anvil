#!/bin/bash

#$ -N anvil-train
#$ -cwd
#$ -l h_vmem=8G
#$ -l h_rt=12:00:00
#$ -pe sharedmem 4

source /etc/profile.d/modules.sh
module load anaconda/5.3.1
source activate anvil-dev

