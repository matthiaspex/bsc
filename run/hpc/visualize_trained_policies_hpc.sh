#!/bin/bash

export MUJOCO_GL="egl"
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True"
# export XLA_PYTHON_CLIENT_MEM_FRACTION="0.99" # jax GPU memory preallocation in %

# set to root directory (BSC folder)
cd "$HOME/bsc/"

# Setup anaconda environment
conda init
source ~/.bashrc
conda activate hope

echo "which python"
which python

export PYTHONPATH="$HOME/bsc/:$HOME/bsc/bsc_utils/"


export POLICY_PARAMS_DIR="./post_processing/trained_policy_params/b11/"
export VIDEO_DIR="$VSC_DATA/brittle_star/tmp/"
export IMAGE_DIR="$VSC_DATA/brittle_star/tmp/"
export RUN_NAME="b11_r01 distance arms [5, 5, 5, 5, 5] popsize 6912 torque reward _ cost hebbian True"

echo "run started"
python "./python_scripts/visualize_trained_policies.py"
echo "run finished"




