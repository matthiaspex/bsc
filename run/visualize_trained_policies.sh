#!/bin/bash

# change to the root directory (BSC)
cd 'C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\'
# single ' important to escape the spaces

source 'C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\bsc\Scripts\activate'
export PYTHONPATH='C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\'



export VIDEO_DIR='C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\Videos\'
export IMAGE_DIR='C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\Images\'
export POLICY_PARAMS_DIR='C:\Users\Matthias\OneDrive - UGent\Documents\DOCUMENTEN\3. Thesis\BSC\trained_policy_params\b03\'
export RUN_NAME='b03_r01 distance arms [5, 5, 5, 5, 5] popsize 400 torque reward _ cost'


python 'visualize_experiments\visualize_trained_policies.py' 



