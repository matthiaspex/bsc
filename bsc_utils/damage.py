# module for damaging morphology
import numpy as np
from jax import numpy as jnp


def check_damage(arm_setup, arm_setup_damage):
    # checks whether a morphology has been damaged
    damage = False
    no_added_segments = True
    for i in range(len(arm_setup)):
        if arm_setup[i] > arm_setup_damage[i]:
            damage = True
        elif arm_setup[i] < arm_setup_damage[i]:
            no_added_segments = False
    assert damage, "None of the arms have lost segments"
    assert no_added_segments, "Some arms have more segments"


def pad_sensory_input(sensory_input, arm_setup, arm_setup_damage, sensor_selection):
    """
    This function is compatible with multiple parallel MJX environments.
    """
    sensory_input_pad = sensory_input
    check_damage(arm_setup, arm_setup_damage)
    segm_sensors_2 = ['joint_position', 'joint_velocity', 'joint_actuator_force'] # every segment has 2 sensors: ip and oop
    segm_sensors_1 = ['segment_contact'] # every segment has 1 sensor
    body_sensors = ['disk_position', 'disk_rotation', 'disk_linear_velocity', 'disk_angular_velocity'] # sensors are related to body

    track_pos = 0

    for type in sensor_selection:
        if type in segm_sensors_2:
            padding_per_segment = 2
        elif type in segm_sensors_1:
            padding_per_segment = 1
        elif type in body_sensors:
            padding_per_segment = 0
        else:
            raise Warning("New sensory inputs not specified in: ['joint_position', 'joint_velocity', 'joint_actuator_force', 'segment_contact', 'disk_position', 'disk_rotation', 'disk_linear_velocity', 'disk_angular_velocity']")

           
        for arm in range(len(arm_setup)):
            if padding_per_segment != 0:
                track_pos += arm_setup_damage[arm] * padding_per_segment
            elif padding_per_segment == 0:
                track_pos += 3
                
            num_segm_rm = arm_setup[arm] - arm_setup_damage[arm]
            if num_segm_rm != 0:
                for i in range(num_segm_rm * padding_per_segment):
                    sensory_input_pad = jnp.insert(sensory_input_pad, track_pos, 0, axis=1)
                    track_pos += 1


    return sensory_input_pad
                    
            


def select_actuator_output(action, arm_setup, arm_setup_damage):
    """
    This function is compatible with multiple parallel MJX environments.
    """
    check_damage(arm_setup, arm_setup_damage)

    track_pos = 0

    for arm in range(len(arm_setup)):
        num_segm_rm = arm_setup[arm] - arm_setup_damage[arm]
        num_segm_maintained = arm_setup[arm] - num_segm_rm
        num_actuators_per_segment = 2


        try:
            action_selection
        except NameError:
            action_selection = action[:, track_pos:track_pos+num_segm_maintained*num_actuators_per_segment]
        else:
            action_selection = jnp.concatenate([action_selection, action[:, track_pos:track_pos+num_segm_maintained*num_actuators_per_segment]], axis = 1)
        
        track_pos += num_segm_maintained * num_actuators_per_segment
        track_pos += num_segm_rm * 2

    return action_selection
    
    
        