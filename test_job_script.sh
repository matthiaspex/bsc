#!/bin/bash

#PBS -N test_job
#PBS -l walltime=0:10:00
#PBS -l nodes=1:ppn:1
#PBS -m abe

# set working directory to directory where you run job script
cd $PBS_O_WORKDIR

echo $PBS_O_WORKDIR

# Setup anaconda environment
conda init
source ~/.bashrc
conda activate hope

export PYTHONPATH="brittle_star:$HOME/bsc/"

# Update code
git pull --recurse-submodules

# Setup wandb
wandb_api_key=`cat $HOME/wandb_key.txt`
export WANDB_API_KEY=$wandb_api_key

# variables
temp="273"

ml load FFmpeg/6.0-GCCcore-12.3.0



echo "Run started"

python Centralized_controller/Test_jobs_files.py 273

echo "Run finished"