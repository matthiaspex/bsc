#!/bin/bash

# change to the root directory (BSC)
cd 'C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\'
# single ' important to escape the spaces

source 'C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\bsc\Scripts\activate'


### arguments
# main arguments
interface='VSC' # either VSC or HPC
arm_setup='5,0,5,0,0' # number of segments per arm, specify for 5 arms
reward_type='distance' # choose from 'distance', 'target' or 'light'

es_popsize='20'
num_generations='20'

sensor_selection='joint_position,joint_actuator_force,segment_contact'
layers='128,128' # number of nodes in the hidden layers only, specify number of layers in the above variable
damage='False'

# wandb
project='Switch_to_VSC' # no spaces
notes='run_2' # no spaces

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
python 'Centralized_controller\centr_ctrl_biomimetic.py' \
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