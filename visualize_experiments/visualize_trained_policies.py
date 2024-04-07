import os
import jax
from jax import numpy as jnp

from evosax import ParameterReshaper

from bsc_utils.miscellaneous import load_config_from_yaml
from bsc_utils.analyze.episode import Simulator
from bsc_utils.controller import NNController

rng = jax.random.PRNGKey(0)

# NOTICE: RUNNING BOTH SIMULATIONS ON LARGE MEMORY GIVES PROBLEMS SAVING THE IMAGES     

VIDEO_DIR = os.environ["VIDEO_DIR"]
IMAGE_DIR = os.environ["IMAGE_DIR"]
POLICY_PARAMS_DIR = os.environ["POLICY_PARAMS_DIR"]
RUN_NAME = os.environ["RUN_NAME"]

trained_policy_params_flat = jnp.load(POLICY_PARAMS_DIR + RUN_NAME + ".npy")
config = load_config_from_yaml(POLICY_PARAMS_DIR + RUN_NAME + ".yaml")

config["damage"]["arm_setup_damage"] = [5,0,5,5,5]
config["environment"]["render"] = {"render_size": [ 3072, 4069 ], "camera_ids": [ 0 ]} # only top down camera
config["evolution"]["penal_expr"] = "nopenal"
config["evolution"]["efficiency_expr"] = config["evolution"]["fitness_expr"]

simulator = Simulator(config)
simulator.generate_env()
simulator.generate_env_damaged()
observation_space_dim, actuator_space_dim = simulator.get_observation_action_space_info()
print(f"""
observation_space_dim = {observation_space_dim}
actuator_space_dim = {actuator_space_dim}
""")

nn_controller = NNController(simulator)
nn_controller.update_model()
policy_params_example = nn_controller.get_policy_params_example()

param_reshaper = ParameterReshaper(policy_params_example) # takes example pytree to know how to reshape pytrees

simulator.update_policy_params_flat(trained_policy_params_flat)
simulator.update_param_reshaper(param_reshaper)
simulator.update_nn_controller(nn_controller)

# print("simulation of single episode started: Undamaged")
# rng, rng_episode = jax.random.split(rng, 2)
# simulator.generate_episode_data_undamaged(rng_episode)
# print("simulation of single episode finished: Undamaged")

# reward = simulator.get_episode_reward()
# cost  = simulator.get_episode_cost()
# penalty = simulator.get_episode_penalty()
# efficiency = simulator.get_episode_efficiency()
# fitness = simulator.get_episode_fitness()
# simulator.get_ip_oop_joint_angles_plot(file_path = IMAGE_DIR + "test joint " + RUN_NAME + ".png")
# simulator.get_increasing_opacity_image(number_of_frames=8, file_path=IMAGE_DIR + "test opacity 8 " + RUN_NAME + ".png")
# simulator.get_episode_video(file_path = VIDEO_DIR + "test" + RUN_NAME + ".mp4", playback_speed=0.5)


# print(f"""
# reward = {reward}
# cost = {cost}
# penalty = {penalty}
# efficiency = {efficiency}
# fitness = {fitness}
# """)

print("simulation of single episode started: Damaged")
rng, rng_episode = jax.random.split(rng, 2)
simulator.generate_episode_data_damaged(rng_episode)
print("simulation of single episode finished: Damaged")

reward_damage = simulator.get_episode_reward()
cost_damage  = simulator.get_episode_cost()
penalty_damage = simulator.get_episode_penalty()
efficiency_damage = simulator.get_episode_efficiency()
fitness_damage = simulator.get_episode_fitness()
simulator.get_ip_oop_joint_angles_plot(file_path = IMAGE_DIR + "test joint DAMAGE " + RUN_NAME + ".png")
simulator.get_increasing_opacity_image(number_of_frames=8, file_path=IMAGE_DIR + "test opacity DAMAGE [5,0,5,5,5] " + RUN_NAME + ".png")
simulator.get_episode_video(file_path = VIDEO_DIR + "test DAMAGE " + RUN_NAME + ".mp4", playback_speed=0.5)


print(f"""
reward_damage = {reward_damage}
cost_damage = {cost_damage}
penalty_damage = {penalty_damage}
efficiency_damage = {efficiency_damage}
fitness_damage = {fitness_damage}
""")

# print(f"""
# reward = {reward} - reward_damage = {reward_damage}
# cost = {cost} - cost_damage = {cost_damage}
# penalty = {penalty} - penalty_damage = {penalty_damage}
# efficiency = {efficiency} - efficiency_damage = {efficiency_damage}
# fitness = {fitness} - fitness_damage = {fitness_damage}
# """)


simulator.clear_envs()
