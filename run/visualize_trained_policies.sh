#!/bin/bash

# set to root directory (BSC folder)
cd "$HOME/OneDrive/Documents/DOCUMENTEN/4_PhD/BSC"

# activate environment via bash in VSC
source "./bsc/bin/activate"

# Not sure whether either of the following lines is required, as the current directory is already correct
export PYTHONPATH=. # relative path, because cd already put the python path to the right location
# export PYTHONPATH="$HOME/OneDrive/Documents/DOCUMENTEN/4_PhD/BSC" (absolute path)


# Set system variables required by the python script
export VIDEO_DIR="./Videos/"
export IMAGE_DIR="./Images/"
export POLICY_PARAMS_DIR="./trained_policy_params/b16/"
export RUN_NAME="b16_r01 target [5, 5, 5, 5, 5] popsize 6912 nocost reward hebbian True"


python "./visualize_experiments/visualize_trained_policies.py"


# Old bash script for windows
: << COMMENT
# change to the root directory (BSC)
cd 'C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\'
# single ' important to escape the spaces

# bsc\Scripts\activate
source 'C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\bsc\Scripts\activate'
export PYTHONPATH='C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\'



export VIDEO_DIR='C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\Videos\'
export IMAGE_DIR='C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\Images\'
export POLICY_PARAMS_DIR='C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\trained_policy_params\b16\'
export RUN_NAME='b16_r01 target [5, 5, 5, 5, 5] popsize 6912 nocost reward hebbian True'


python 'visualize_experiments\visualize_trained_policies.py' 

COMMENT




