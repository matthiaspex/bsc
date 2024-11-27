#!/bin/bash

#PBS -N batch_20_run_1_donphan
#PBS -l nodes=1:ppn=2
# #PBS -l gpus=1 # not necessary to specify on donphan, but DON'T FORGET on ACCELGOR
#PBS -l mem=12gb
#PBS -l walltime=72:00:00
#PBS -e /user/gent/457/vsc45787/bsc/Job_scripts_logs
#PBS -o /user/gent/457/vsc45787/bsc/Job_scripts_logs
#PBS -m abe

ml purge
ml load glew/2.2.0-GCCcore-12.3.0-osmesa
ml load FFmpeg/6.0-GCCcore-12.3.0
# ml load typing-extensions/3.10.0.2-GCCcore-11.2.0
export MUJOCO_GL="egl"
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True"
# export XLA_PYTHON_CLIENT_MEM_FRACTION="0.90" # max jax GPU memory allocation %


# Setup wandb
wandb_api_key=`cat $HOME/wandb_key.txt`
export WANDB_API_KEY=$wandb_api_key

# set working directory to directory where you run job script
cd "$HOME/bsc/"
pwd

# Setup anaconda environment
conda init
source ~/.bashrc
conda activate hope

echo "which python"
which python

export PYTHONPATH="$HOME/bsc/:$HOME/bsc/bsc_utils/"
export POLICY_PARAMS_DIR="./post_processing/trained_policy_params/"
export VIDEO_DIR="$VSC_DATA/brittle_star/tmp/"
export IMAGE_DIR="$VSC_DATA/brittle_star/tmp/"

# # Update code
# git pull --recurse-submodules



echo "Run started"
export CONFIG_FILE="./config/decentralized_donphan_job.yaml"
python './python_scripts/decentralized_hebbian_training.py' 
echo "Run finished"


rm $VIDEO_DIR*
# rm $IMAGE_DIR* # Only remove when it is actually different from the $VIDEO_DIR

