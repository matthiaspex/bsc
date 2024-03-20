#!/bin/bash

#PBS -N test_job
#PBS -l walltime=0:10:00
#PBS -l nodes=1:ppn=1
#PBS -m abe
#PBS -e Job_scripts_logs/
#PBS -o Job_scripts_logs/

# set working directory to directory where you run job script
cd $PBS_O_WORKDIR

echo $PBS_O_WORKDIR


# Setup anaconda environment
conda init
source ~/.bashrc
conda activate hope

echo "which python"
which python

export PYTHONPATH="$HOME/bsc/"

# Update code
git pull --recurse-submodules

# Setup wandb
wandb_api_key=`cat $HOME/bsc/wandb_key.txt`
export WANDB_API_KEY=$wandb_api_key

# variables
# temp="273"

ml load FFmpeg/6.0-GCCcore-12.3.0
ml load typing-extensions/4.9.0-GCCcore-12.3.0


echo "Run started"

# python Centralized_controller/Test_jobs_files.py $temp

# python Centralized_controller/centr_ctrl_biomimetic.py

/user/gent/457/vsc45787/.conda/envs/hope/bin/python Centralized_controller/centr_ctrl_biomimetic.py

echo "Run finished"