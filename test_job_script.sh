#!/bin/bash

#PBS -N test_job
#PBS -l nodes=1:ppn=2
# #PBS -l gpus=1
#PBS -l mem=8gb
#PBS -l walltime=0:10:00
#PBS -e Job_scripts_logs/
#PBS -o Job_scripts_logs/
#PBS -m abe

ml purge
ml load glew/2.2.0-GCCcore-11.2.0-osmesa
ml load FFmpeg/4.3.2-GCCcore-11.2.0
# ml load typing-extensions/3.10.0.2-GCCcore-11.2.0
export MUJOCO_GL="osmesa"
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True"

export TMP_DIR="$VSC_DATA/brittle_star/tmp"


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

export PYTHONPATH="$HOME/bsc/:$HOME/bsc/bsc_utils"


# Update code
git pull --recurse-submodules


echo "Run started"

# python Centralized_controller/Test_jobs_files.py $temp

# python Centralized_controller/centr_ctrl_biomimetic.py

# /user/gent/457/vsc45787/.conda/envs/hope/bin/python Centralized_controller/centr_ctrl_biomimetic.py


### arguments
# main arguments
interface='HPC' # either VSC or HPC
arm_setup='5,0,5,0,0' # number of segments per arm, specify for 5 arms
reward_type='distance' # choose from 'distance', 'target' or 'light'

es_popsize='20'
num_generations='20'

sensor_selection='joint_position,joint_actuator_force,segment_contact'
layers='128,128' # number of nodes in the hidden layers only, specify number of layers in the above variable
damage='False'

# wandb
project='Switch_to_VSC' # no spaces
notes='run_3' # no spaces

# optional arguments: if not provided, default values can be found in the python script itself
joint_control='position' # either position or torque
sand_ground_color='False'
num_physics_steps_per_control_step='10'
simulation_time='5'
joint_randomization_noise_scale='0.0'
light_perlin_noise_scale='4' # is only used in case of 'light' reward type
target_distance='3' # is only used in case of 'target' reward type
arm_setup_damage=$arm_setup # Will only be used if the $damage variable is set to 'True'



# Run the Python script
# python 'Centralized_controller/centr_ctrl_biomimetic.py' \
python 'Centralized_controller/centr_ctrl_biomimetic.py' \
  $interface \
  $arm_setup \
  $reward_type \
  $es_popsize \
  $num_generations \
  $sensor_selection \
  $layers \
  $damage \
  $project \
  $notes \
#   --joint_control $actuator_control \
#   --sand_ground_color $sand_ground_color \
#   --num_physics_steps_per_control_step $num_physics_steps_per_control_step \
#   --simulation_time $simulation_time \
#   --joint_randomization_nosie_scale $joint_randomization_noise_scale \
#   --light_perlin_noise_scale $light_perlin_noise_scale \
#   --target_distance $target_distance \
#   --arm_setup_damage $arm_setup_damage \

echo "Run finished"