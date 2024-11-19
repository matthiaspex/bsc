#!/bin/bash

# set to root directory (BSC folder)
cd "$HOME/OneDrive/Documents/DOCUMENTEN/4_PhD/BSC"

# activate environment via bash in VSC
source "./bsc/bin/activate"

# Not sure whether either of the following lines is required, as the current directory is already correct
export PYTHONPATH=. # relative path, because cd already put the python path to the right location
# export PYTHONPATH="$HOME/OneDrive/Documents/DOCUMENTEN/4_PhD/BSC" (absolute path)

export MUJOCO_GL="egl"

# Set system variables required by the python script
export CONFIG_FILE="./config/general_bsc.yaml"
export VIDEO_DIR="./post_processing/Videos/tmp/"
export IMAGE_DIR="./post_processing/Images/tmp/"
export POLICY_PARAMS_DIR="./post_processing/trained_policy_params/tmp/"


python "./python_scripts/decentralized_hebbian_training.py"



# Old bash script for windows
: << COMMENT
# change to the root directory (BSC)
cd 'C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\'
# single ' important to escape the spaces

source 'C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\bsc\Scripts\activate'
export PYTHONPATH='C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\'

export CONFIG_FILE='C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\config\general_bsc.yaml'
export VIDEO_DIR='C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\Videos\'
export IMAGE_DIR='C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\Images\'
export POLICY_PARAMS_DIR='C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\trained_policy_params\'

python 'Hebbian_centr_ctrl\centralized_hebbian_training.py' 

COMMENT