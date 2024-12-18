import os
import sys

import jax
from jax import numpy as jnp

from bsc_utils.controller.decentralized import DecentralisedController
from bsc_utils.miscellaneous import load_config_from_yaml, complete_config_with_defaults, get_target_positions
from bsc_utils.simulate.analyze import Simulator
from bsc_utils.controller.base import NNController, ExplicitMLP
from bsc_utils.controller.hebbian import HebbianController


rng = jax.random.PRNGKey(0)

# NOTICE: RUNNING BOTH SIMULATIONS ON HIGH RESOLUTION GIVES PROBLEMS SAVING THE IMAGES AND VIDEOS     

VIDEO_DIR = os.environ["VIDEO_DIR"]
IMAGE_DIR = os.environ["IMAGE_DIR"]
POLICY_PARAMS_DIR = os.environ["POLICY_PARAMS_DIR"]
RUN_NAME = os.environ["RUN_NAME"]


trained_policy_params_flat = jnp.load(POLICY_PARAMS_DIR + RUN_NAME + ".npy")
parallel_dim = trained_policy_params_flat.shape[0]
config = load_config_from_yaml(POLICY_PARAMS_DIR + RUN_NAME + ".yaml")

# especially relevant when running older experiments that might not have certain required arguments in their setup
config = complete_config_with_defaults(config)

####################################################################################
# # finutune episode simulation

# simulation_time = 10
# config["environment"]["simulation_time"] = simulation_time
playback_speed = 1

# config["morphology"]["replace_joint_stiffness"] = (True, 0.1)
# config["morphology"]["replace_joint_damping"] = (True, 0.5)
# config["morphology"]["replace_joint_armature"] = (True, 0.02)

simulate_undamaged = True
simulate_damaged = True

joint_angle_plots = False
opacity_frames_image = False
video_render = False
kernel_animation = True
kernel_animation_arm_selection = [0,1,2,3,4] # in case of 1 arm, it can be int or list of 1 int
kernel_histogram = True
synapse_time_evolution = True
specific_synapses =[("embed", 0, 1, 20, 20)]
num_random_synapses = 20
lr_histogram = False

config["training"]["target"]["force_single_direction"] = (True, "rowing", 1)

arm_setup_damage = [5,5,0,5,5]
config["damage"]["arm_setup_damage"] = arm_setup_damage
config["arena"]["sand_ground_color"] = False
config["environment"]["render"] = {"render_size": [ 480, 640 ], "camera_ids": [ 0, 1 ]} # only static aquarium camera camera [ 0 ], otherwise: "camera_ids": [ 0, 1 ]
                            # choose ratio 3:4 --> [ 480, 640 ], [ 720, 960 ], [ 960, 1280 ] (720p), [ 1440, 1920 ] (1080p), [ 3072, 4069 ] (HDTV 4k)


run_name_addition = ""
# if config["arena"]["sand_ground_color"] == False:
#     run_name_addition += " no sand"

if playback_speed != 1:
    run_name_addition += f" x{playback_speed}"

if config["environment"]["simulation_time"] != 5:
    run_name_addition += f" {config["environment"]["simulation_time"]}s"

if config["training"]["target"]["force_single_direction"][0] == True:
    run_name_addition += f" target {config["training"]["target"]["force_single_direction"][1]} {config["training"]["target"]["force_single_direction"][2]}"

####################################################################################



simulator = Simulator(config)
simulator.generate_env()
simulator.generate_env_damaged()
observation_space_dim, actuator_space_dim = simulator.get_observation_action_space_info()
print(f"""
observation_space_dim = {observation_space_dim}
actuator_space_dim = {actuator_space_dim}
""")

if config["environment"]["reward_type"] == 'target':
    rng, rng_targets_simulator = jax.random.split(rng, 2)
    targets_simulator = get_target_positions(rng=rng_targets_simulator,
                                    distance=config["training"]["target"]["distance"],
                                    num_rowing=0,
                                    num_reverse_rowing=0,
                                    num_random_positions=1,
                                    parallel_dim=trained_policy_params_flat.shape[0],
                                    parallel_constant=True,
                                    force_single_direction=config["training"]["target"]["force_single_direction"])
    simulator.update_targets(targets_simulator[0])

if config["controller"]["decentralized"]["decentralized_on"] == True:
    nn_controller = DecentralisedController(simulator, parallel_dim=parallel_dim)
    if config["controller"]["hebbian"] == True:
        nn_controller.update_model(controller=HebbianController) # all information to build controller present in EnvContainer (hidden layers, action dim, obs dim)
    else:
        nn_controller.update_model(controller=NNController)

elif config["controller"]["decentralized"]["decentralized_on"] == False:
    if config["controller"]["hebbian"] == True:
        nn_controller = HebbianController(simulator)
    else:
        nn_controller = NNController(simulator)
    
    nn_controller.update_model(ExplicitMLP)

nn_controller.update_parameter_reshaper() # as the model is already defined and the environmetns are available from the environment container and config files in simulator in the simulator
# param_reshaper = nn_controller.get_parameter_reshaper() # --> if you would want to do stuff with this parameter_reshaper.

nn_controller.update_policy_params(trained_policy_params_flat)
simulator.update_nn_controller(nn_controller)

if lr_histogram == True:
    simulator.get_learning_rule_histograms(file_path = IMAGE_DIR + RUN_NAME + run_name_addition + " LR HISTOGRAM.png",\
                                             xlabel="parameter value",\
                                             title="Learning rule parameters distributions")

if simulate_undamaged:
    print("simulation of single episode started: Undamaged")
    rng, rng_episode = jax.random.split(rng, 2)
    simulator.generate_episode_data_undamaged(rng_episode)
    print("simulation of single episode finished: Undamaged")

    reward = simulator.get_episode_reward()
    cost  = simulator.get_episode_cost()
    penalty = simulator.get_episode_penalty()
    efficiency = simulator.get_episode_efficiency()
    fitness = simulator.get_episode_fitness()
    if joint_angle_plots == True:
        simulator.get_ip_oop_joint_angles_plot(file_path = IMAGE_DIR + RUN_NAME + run_name_addition + "PLOTS.png")
    if opacity_frames_image == True:
        simulator.get_increasing_opacity_image(number_of_frames=8, file_path=IMAGE_DIR + RUN_NAME + run_name_addition + " OPACITY.png")
    if video_render == True:
        simulator.get_episode_video(file_path = VIDEO_DIR + RUN_NAME + run_name_addition + ".mp4", playback_speed=playback_speed)
    if kernel_animation == True:
        if config["controller"]["decentralized"]["decentralized_on"] == True:
            simulator.get_decentralized_kernel_animation(file_path=VIDEO_DIR + "kernel " + RUN_NAME + run_name_addition + ".mp4",
                                                        arm_selection=kernel_animation_arm_selection)
        else:
            simulator.get_kernel_animation(file_path = VIDEO_DIR + "kernel " + RUN_NAME + run_name_addition + ".mp4")
    if kernel_histogram == True:
        simulator.get_final_kernel_histogram(file_path = IMAGE_DIR + "histogram " + RUN_NAME + run_name_addition + ".png",\
                                             xlabel="synapse weights",\
                                             title="Final weight distribution - Undamaged")
    if synapse_time_evolution == True:
        simulator.get_synapse_time_evolutions(file_path=IMAGE_DIR + RUN_NAME + run_name_addition + " synapse plots.png",
                                              rng=rng,
                                              specific_synapses=specific_synapses,
                                              num_random_synapses=num_random_synapses)
    



    print(f"""
    reward = {reward}
    cost = {cost}
    penalty = {penalty}
    efficiency = {efficiency}
    fitness = {fitness}
    """)

if simulate_damaged:
    print("simulation of single episode started: Damaged")
    rng, rng_episode = jax.random.split(rng, 2)
    simulator.generate_episode_data_damaged(rng_episode)
    print("simulation of single episode finished: Damaged")

    reward_damage = simulator.get_episode_reward()
    cost_damage  = simulator.get_episode_cost()
    penalty_damage = simulator.get_episode_penalty()
    efficiency_damage = simulator.get_episode_efficiency()
    fitness_damage = simulator.get_episode_fitness()
    if joint_angle_plots == True:
        simulator.get_ip_oop_joint_angles_plot(file_path = IMAGE_DIR + RUN_NAME + run_name_addition + " PLOTS DAMAGE.png")
    if opacity_frames_image == True:
        simulator.get_increasing_opacity_image(number_of_frames=8, file_path=IMAGE_DIR + RUN_NAME + run_name_addition + "OPACITY DAMAGE.png")
    if video_render == True:
        simulator.get_episode_video(file_path = VIDEO_DIR + RUN_NAME + run_name_addition + " DAMAGE.mp4", playback_speed=playback_speed)
    if kernel_animation == True:
        if config["controller"]["decentralized"]["decentralized_on"] == True:
            simulator.get_decentralized_kernel_animation(file_path=VIDEO_DIR + "kernel " + RUN_NAME + run_name_addition + " DAMAGE.mp4",
                                                        arm_selection=kernel_animation_arm_selection)
        else:
            simulator.get_kernel_animation(file_path=VIDEO_DIR + "kernel " + RUN_NAME + run_name_addition + " DAMAGE.mp4")
    if kernel_histogram == True:
        simulator.get_final_kernel_histogram(file_path=IMAGE_DIR + "histogram " + RUN_NAME + run_name_addition + " DAMAGE.png",\
                                             xlabel="synapse weights",\
                                             title="Final weight distribution - Damaged")
    if synapse_time_evolution == True:
        simulator.get_synapse_time_evolutions(file_path=IMAGE_DIR + RUN_NAME + run_name_addition + " synapse plots DAMAGE.png",
                                              rng=rng,
                                              specific_synapses=specific_synapses,
                                              num_random_synapses=num_random_synapses)        


    print(f"""
    reward_damage = {reward_damage}
    cost_damage = {cost_damage}
    penalty_damage = {penalty_damage}
    efficiency_damage = {efficiency_damage}
    fitness_damage = {fitness_damage}
    """)

if simulate_undamaged and simulate_damaged:
    print(f"""
    reward = {reward} - reward_damage = {reward_damage}
    cost = {cost} - cost_damage = {cost_damage}
    penalty = {penalty} - penalty_damage = {penalty_damage}
    efficiency = {efficiency} - efficiency_damage = {efficiency_damage}
    fitness = {fitness} - fitness_damage = {fitness_damage}
    """)

simulator.clear_envs()
