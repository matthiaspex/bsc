#!/bin/bash

#PBS -N batch_02
#PBS -l nodes=1:ppn=2
#PBS -l gpus=1
#PBS -l mem=8gb
#PBS -l walltime=5:00:00
#PBS -e Job_scripts_logs/
#PBS -o Job_scripts_logs/
#PBS -m abe

ml purge
ml load glew/2.2.0-GCCcore-11.2.0-osmesa
ml load FFmpeg/4.3.2-GCCcore-11.2.0
# ml load typing-extensions/3.10.0.2-GCCcore-11.2.0
export MUJOCO_GL="osmesa"
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True"



# Setup wandb
wandb_api_key=`cat $HOME/wandb_key.txt`
export WANDB_API_KEY=$wandb_api_key

# set working directory to directory where you run job script
cd $PBS_O_WORKDIR

echo $PBS_O_WORKDIR

# Setup anaconda environment
conda init
source ~/.bashrc
conda activate hope

echo "which python"
which python

export PYTHONPATH="$HOME/bsc/:$HOME/bsc/bsc_utils/"
export POLICY_PARAMS_DIR="$HOME/bsc/trained_policy_params/"
export VIDEO_DIR="$VSC_DATA/brittle_star/tmp/"


# Update code
git pull --recurse-submodules



for file in $HOME/bsc/config/batch/*.yaml
do
  echo "Run started"
  echo $file
  export CONFIG_FILE="$file"
  python 'Centralized_controller/centr_ctrl_biomimetic.py' 
  echo "Run finished"
done

rm $VIDEO_DIR*

