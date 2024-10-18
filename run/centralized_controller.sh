#!/bin/bash

# set to root directory (BSC folder)
cd "$HOME/OneDrive/Documents/DOCUMENTEN/4_PhD/BSC"

# activate environment via bash in VSC
source "./bsc/bin/activate"

# Not sure whether either of the following lines is required, as the current directory is already correct
export PYTHONPATH=. # relative path, because cd already put the python path to the right location
# export PYTHONPATH="$HOME/OneDrive/Documents/DOCUMENTEN/4_PhD/BSC" (absolute path)


# Set system variables required by the python script
export CONFIG_FILE="./config/general_bsc.yaml"
export VIDEO_DIR="./Videos/"
export IMAGE_DIR="./Images/"
export POLICY_PARAMS_DIR="./trained_policy_params/"


python "./Centralized_controller/centr_ctrl_biomimetic.py"




: << COMMENT
# change to the root directory (BSC)
cd 'C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\'
# single ' important to escape the spaces

source 'C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\bsc\Scripts\activate'


export CONFIG_FILE='C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\config\general_bsc.yaml'
export VIDEO_DIR='C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\Videos\'
export POLICY_PARAMS_DIR='C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\trained_policy_params\'
export PYTHONPATH='C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\'

python 'Centralized_controller\centr_ctrl_biomimetic_updated.py' 

COMMENT

