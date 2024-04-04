import yaml
import sys
import os

print(f"sys.executable: {sys.executable}")

import numpy as np
import jax
from jax import numpy as jnp
import evosax
from evosax import OpenES, ParameterReshaper, NetworkMapper, FitnessShaper
import flax
from flax import linen as nn
from typing import Any, Callable, Sequence, Union, List
import biorobot
import time
import matplotlib.pyplot as plt

import wandb

import moviepy
import imageio
import plotly
import cv2
import mediapy as media






rng = jax.random.PRNGKey(0) # make an rng right away and every split throughout the document should make a new rng
# this new rng should only be used for the sole purpose of splitting in the future

# custom modules import --> from bsc_utils directory
from bsc_utils.visualization import visualize_mjcf, show_video, create_video, post_render, plot_ip_oop_joint_angles 
from bsc_utils.controller import ExplicitMLP
from bsc_utils.BrittleStarEnv import create_morphology, create_arena, create_environment, full_mjcf_configurations
from bsc_utils.damage import check_damage, pad_sensory_input, select_actuator_output
from bsc_utils.simulation import rollout
rollout = jax.jit(rollout, static_argnames=("mjx_vectorized_env", "nn_model",  "total_num_control_timesteps", "NUM_MJX_ENVIRONMENTS", "sensor_selection", "cost_expr", "target_position"))
from bsc_utils.miscallaneous import check_GPU_access
from bsc_utils.evolution import reward_cost_to_fitness


# some specific inports from biorobot
from biorobot.brittle_star.mjcf.morphology.specification.default import default_brittle_star_morphology_specification
from biorobot.brittle_star.environment.light_escape.shared import BrittleStarLightEscapeEnvironmentConfiguration
from biorobot.brittle_star.environment.directed_locomotion.shared import \
    BrittleStarDirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.environment.undirected_locomotion.shared import \
    BrittleStarUndirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.mjcf.arena.aquarium import AquariumArenaConfiguration, MJCFAquariumArena




np.set_printoptions(precision=3, suppress=False, linewidth=100)
jnp.set_printoptions(precision=3, suppress=False, linewidth=100)

###############################################
# READOUT ALL ARGUMENTS FROM YAML/SYS VARIABLES
###############################################

VIDEO_DIR = os.environ["VIDEO_DIR"]
POLICY_PARAMS_DIR = os.environ["POLICY_PARAMS_DIR"]
CONFIG_FILE = os.environ["CONFIG_FILE"]


print(f"""
      CONFIG_FILE path: {CONFIG_FILE}
      POLICY_PARAMS_DIR path: {POLICY_PARAMS_DIR}
      VIDEO_DIR path: {VIDEO_DIR}
""")

with open(CONFIG_FILE, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


# readout the main arguments
# experiment
interface = config["experiment"]["interface"]
run_name_format = config["experiment"]["run_name_format"]
# wandb
project = config["wandb"]["project"]
group = config["wandb"]["group"]
tags = config["wandb"]["tags"]
notes = config["wandb"]["notes"]
# morphology
arm_setup = config["morphology"]["arm_setup"]
joint_control = config["morphology"]["joint_control"]
# arena
arena_size = config["arena"]["arena_size"]
sand_ground_color = config["arena"]["sand_ground_color"]
# environment
reward_type = config["environment"]["reward_type"]
sensor_selection = config["environment"]["sensor_selection"]
num_physics_steps_per_control_step = config["environment"]["num_physics_steps_per_control_step"]
simulation_time = config["environment"]["simulation_time"]
joint_randomization_noise_scale = config["environment"]["joint_randomization_noise_scale"]
light_perlin_noise_scale = config["environment"]["light_perlin_noise_scale"]
target_distance = config["environment"]["target_distance"]
target_position = config["environment"]["target_position"]
color_contacts = config["environment"]["color_contacts"]
# damage
damage = config["damage"]["damage"]
arm_setup_damage = config["damage"]["arm_setup_damage"]
# evolution
es_popsize = config["evolution"]["es_popsize"]
num_generations = config["evolution"]["num_generations"]
cost_expr = config["evolution"]["cost_expr"]
fitness_expr = config["evolution"]["fitness_expr"]
# controller
hidden_layers = config["controller"]["hidden_layers"]

config_flat = {}
for k, v in config.items():
    if not isinstance(v, dict):
        config_flat[k] = v
    else:
        for l, w in v.items():
            if not isinstance(w, dict):
                config_flat[l] = w
        
run_name = run_name_format.format(**config_flat)

print(f"run_name: {run_name}")


check_GPU_access(interface = interface)


morphology_specification, arena_configuration, environment_configuration = full_mjcf_configurations(config["morphology"], config["arena"], config["environment"])

morphology = create_morphology(morphology_specification=morphology_specification)
visualize_mjcf(mjcf=morphology)

arena = create_arena(arena_configuration=arena_configuration)
visualize_mjcf(mjcf=arena)

total_num_control_timesteps = environment_configuration.total_num_control_steps

##########################################
# instantiating the vectorized environment
##########################################
NUM_MJX_ENVIRONMENTS = es_popsize # best to get close to number of CUDA cores available

mjx_vectorized_env = create_environment(
                morphology_specification=morphology_specification,
                arena_configuration=arena_configuration,
                environment_configuration=environment_configuration,
                backend="MJX"
                )

rng, mjx_vectorized_env_rng = jax.random.split(rng, 2)
mjx_vectorized_env_rng = jnp.array(jax.random.split(mjx_vectorized_env_rng, NUM_MJX_ENVIRONMENTS))

# vectorizing the functionalities
mjx_vectorized_step = jax.jit(jax.vmap(mjx_vectorized_env.step))
mjx_vectorized_reset = jax.jit(jax.vmap(mjx_vectorized_env.reset))

# reset the state of all parallel mjx environments
mjx_vectorized_state = mjx_vectorized_reset(mjx_vectorized_env_rng)


sensors = [key for key in mjx_vectorized_state.observations.keys()]
sensors_with_dim = {s: d.shape for s, d in mjx_vectorized_state.observations.items()}
sensor_selection_dim = sum([mjx_vectorized_state.observations[sensor].shape[-1] for sensor in sensor_selection])

print(f"""
      sensors: {sensors}
      sensors with dimensions {sensors_with_dim}
      sensor selection: {sensor_selection}
      sensor selection dimension: {sensor_selection_dim}
""")


sensory_input_nn_test = jnp.concatenate(
    [mjx_vectorized_state.observations[label] for label in sensor_selection],
    axis = 1
)
print(f"""
      sensory_input_nn_test.shape: {sensory_input_nn_test.shape}
""")

nn_input_dim = sensor_selection_dim
nn_output_dim = len(mjx_vectorized_env.actuators)
print(f"""
      Dimension of sensory space: {nn_input_dim}
      Dimension of actuator space: {nn_output_dim}
""")

###################################
# Instantiating the neural network
###################################

features = hidden_layers + [nn_output_dim]
print(f"""
      features = {features}
      hidden_layers = {hidden_layers}
      type(hidden_layers) = {type(hidden_layers)}
      """)
nn_model = ExplicitMLP(features = tuple(features), joint_control = joint_control)


# initialising the parameters of the model
rng, rng_input, rng_init = jax.random.split(rng, 3)
policy_params_init = nn_model.init(rng_init, jax.random.uniform(rng_input, (nn_input_dim,)))

# vectorize the model.apply function
vectorized_nn_model_apply = jax.jit(jax.vmap(nn_model.apply))

# params is a PyTree --> see jax documentation
# print(params)
print(f"""nn features: {features}
      policy_params_init pytree keys: {policy_params_init['params'].keys()}
      policy params pytree description: {jax.tree_util.tree_map(lambda x: x.shape, policy_params_init)}
""")

# single forward pass through the model
test_output = nn_model.apply(policy_params_init,jax.random.uniform(rng_input, (nn_input_dim,)))
print(f"""
      Range of the output is [{jnp.min(test_output)}, {jnp.max(test_output)}]
      This range should be between -0.52 and +0.52 for joint angle control and between -1 and 1 for torques
""")

param_reshaper = ParameterReshaper(policy_params_init)
num_params = param_reshaper.total_params # get from the weights and biases of the NN


###################################
# Instantiating the search strategy
###################################
rng, rng_ask, rng_init = jax.random.split(rng, 3)
strategy  = OpenES(popsize = es_popsize, num_dims = num_params)
# still parameters that can be finetuned, like optimisation method, lrate, lrate decay, ...
es_params = strategy.default_params
# # replacing certain parameters:
# es_params = es_params.replace(init_min = -3, init_max = 3)
print(f"es_params: {es_params}\n")

fit_shaper = FitnessShaper(maximize=True)

es_state = strategy.initialize(rng_init, es_params)

policy_params_test, es_state = strategy.ask(rng_ask, es_state)
print(f"candidate solution shape: {policy_params_test.shape}\n")

# network parameters have an additional dimension of 100, meaning that every bias and every weight appears 100 times
# 100 parallel trees if you want
policy_params_test_reshaped = param_reshaper.reshape(policy_params_test)
print(f"""
      reshaped policy params in jax:
      {jax.tree_util.tree_map(lambda x: x.shape, policy_params_test_reshaped)}
""")



##########################
# RUNNING THE OPTIMISATION
##########################


# reset the search strategy state
rng, rng_init = jax.random.split(rng, 2)
es_state = strategy.initialize(rng_init, es_params)

policy_params_to_render = []

wandb.init(
    project = project,
    group = group,
    name = run_name,
    tags = tags,
    notes = notes,
    config={
    "nn_architecture": [nn_input_dim] + features,
    "arm_setup": arm_setup,
    "joint_control": joint_control,
    "reward_type": reward_type,
    "num_params": num_params,
    "num_generations": num_generations,
    "es_popsize": es_popsize,
    "sensor_selection": sensor_selection,
    "simulation_time": simulation_time,
    "total_num_control_steps": total_num_control_timesteps
    }
)

# Run ask-eval-tell loop
start_time = time.time()
for gen in range(num_generations):
    # track runtimes
    if gen%10 == 0:
        print('generation: ', gen, '\ntime since start: ', time.time()-start_time)
        
    
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    
    policy_params, es_state = strategy.ask(rng_gen, es_state, es_params)
    policy_params_shaped = param_reshaper.reshape(policy_params) # --> stacked pytree


    final_state, rewards, costs, rng = rollout(mjx_vectorized_env = mjx_vectorized_env,
                                               nn_model = nn_model,
                                               policy_params_shaped = policy_params_shaped,
                                               total_num_control_timesteps = total_num_control_timesteps,
                                               sensor_selection = sensor_selection,
                                               rng = rng,
                                               NUM_MJX_ENVIRONMENTS = NUM_MJX_ENVIRONMENTS,
                                               cost_expr = cost_expr
                                               )

    # rewards shape is: e.g. 300 parallel rewards per control step, an additional stack every control step --> sum over the control steps along axis = 0
    total_reward = jnp.sum(rewards, axis = 0)
    total_cost = jnp.sum(costs, axis = 0)
    fitness = reward_cost_to_fitness(total_reward, total_cost, fitness_expr)

    # fitness should be an array with population size as len (e.g. 100)
    fit_re = fit_shaper.apply(policy_params, fitness)

   
    # log metrics to wandb
    wandb.log({"mean reward": jnp.mean(total_reward),
               "max reward":jnp.max(total_reward),
               "mean cost": jnp.mean(total_cost),
               "min cost": jnp.min(total_cost),
               "mean fitness": jnp.mean(fitness),
               "max fitness": jnp.max(fitness)
              })
    
    # # Select certain training generations to render videos
    # if gen in [2, 5, 10, 20, 40, 80]:
    #     policy_params_to_render.append(es_state.best_member)
    
    es_state = strategy.tell(policy_params, fit_re, es_state, es_params)
    


# Get best overall population member
policy_params_to_render.append(policy_params[jnp.argmax(fitness)])
print('Policy training finished!')


jnp.save(POLICY_PARAMS_DIR+run_name+".npy", policy_params_to_render)

#####################################
# Video and angle plots visualisation
#####################################
policy_params_to_render = jnp.array(policy_params_to_render)
policy_params_solution_shaped = param_reshaper.reshape(policy_params_to_render)
print(f"""
Rendered policy params tree shape
{jax.tree_util.tree_map(lambda x: x.shape, policy_params_solution_shaped)}
""")

fps = int(1/environment_configuration.control_timestep)

NUM_MJX_ENVIRONMENTS_render = policy_params_to_render.shape[0]

rng, mjx_vectorized_env_rng = jax.random.split(rng, 2)
mjx_vectorized_env_rng = jnp.array(jax.random.split(mjx_vectorized_env_rng, NUM_MJX_ENVIRONMENTS_render))

# vectorizing the functionalities
mjx_vectorized_step = jax.jit(jax.vmap(mjx_vectorized_env.step))
mjx_vectorized_reset = jax.jit(jax.vmap(mjx_vectorized_env.reset))

mjx_vectorized_state = mjx_vectorized_reset(rng=mjx_vectorized_env_rng)

mjx_frames = []

# joint sensor observations order seg1ip, seg1oop, seg2ip, seg2oop, ...
joint_angles_ip = []
joint_angles_oop = []

i = 0
while not jnp.any(mjx_vectorized_state.terminated | mjx_vectorized_state.truncated):
    if i%10 == 0:
        print(i)
    i += 1
    
    sensory_input = jnp.concatenate(
        [mjx_vectorized_state.observations[label] for label in sensor_selection],
        axis = 1
    )

    joint_angles_ip_t = []
    joint_angles_oop_t = []
    j = 0
    for n in arm_setup:
        if n != 0:
            joint_angles_ip_t.append(mjx_vectorized_state.observations["joint_position"][0][j*2*n:(j+1)*2*n:2])
            joint_angles_oop_t.append(mjx_vectorized_state.observations["joint_position"][0][j*2*n+1:(j+1)*2*n+1:2])
            j += 1

    joint_angles_ip.append(joint_angles_ip_t)
    joint_angles_oop.append(joint_angles_oop_t)

    action = vectorized_nn_model_apply(policy_params_solution_shaped, sensory_input)
    
    mjx_vectorized_state = mjx_vectorized_step(state=mjx_vectorized_state, action=action)
    
    mjx_frames.append(
            post_render(
                mjx_vectorized_env.render(state=mjx_vectorized_state),
                mjx_vectorized_env.environment_configuration
                )
            )

imgio_kargs = {
    'fps': fps, 'quality': 10, 'macro_block_size': None, 'codec': 'h264',
    'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']
    }
writer = imageio.get_writer(VIDEO_DIR + run_name + ".mp4", **imgio_kargs)
for frame in mjx_frames:
    writer.append_data(frame)
writer.close()

wandb.log({"Video trained model": wandb.Video(VIDEO_DIR + run_name + ".mp4",
                                              caption=run_name, fps=fps, format='mp4')})




fig, axes = plot_ip_oop_joint_angles(joint_angles_ip, joint_angles_oop)
wandb.log({"Joint Angles trained model": wandb.Image(fig)})



#########################################################
# Damage: Applying trained policy to a damaged morphology
#########################################################

if damage:
    check_damage(arm_setup = arm_setup, arm_setup_damage = arm_setup_damage)
    # specifying morphology
    morphology_specification_damage = default_brittle_star_morphology_specification(
            num_arms=len(arm_setup_damage), num_segments_per_arm=arm_setup_damage, use_p_control=(joint_control == 'position'), use_torque_control=(joint_control == 'torque')
            )
    morphology_damage = create_morphology(morphology_specification=morphology_specification_damage)
    visualize_mjcf(mjcf=morphology_damage)

    mjx_vectorized_env = create_environment(
                    morphology_specification=morphology_specification_damage,
                    arena_configuration=arena_configuration,
                    environment_configuration=environment_configuration,
                    backend="MJX"
                    )



    NUM_MJX_ENVIRONMENTS_render = policy_params_to_render.shape[0]

    rng, mjx_vectorized_env_rng = jax.random.split(rng, 2)
    mjx_vectorized_env_rng = jnp.array(jax.random.split(mjx_vectorized_env_rng, NUM_MJX_ENVIRONMENTS_render))

    # vectorizing the functionalities
    mjx_vectorized_step = jax.jit(jax.vmap(mjx_vectorized_env.step))
    mjx_vectorized_reset = jax.jit(jax.vmap(mjx_vectorized_env.reset))

    mjx_vectorized_state = mjx_vectorized_reset(rng=mjx_vectorized_env_rng)



    mjx_frames_damage = []

    # joint sensor observations order seg1ip, seg1oop, seg2ip, seg2oop, ...
    joint_angles_ip = []
    joint_angles_oop = []

    i = 0
    while not jnp.any(mjx_vectorized_state.terminated | mjx_vectorized_state.truncated):
        if i%10 == 0:
            print(i)
        i += 1
        
        sensory_input = jnp.concatenate(
            [mjx_vectorized_state.observations[label] for label in sensor_selection],
            axis = 1
        )

        sensory_input_pad = pad_sensory_input(sensory_input, arm_setup, arm_setup_damage, sensor_selection)
        
        joint_angles_ip_t = []
        joint_angles_oop_t = []
        j = 0
        for n in arm_setup_damage:
            if n != 0:
                joint_angles_ip_t.append(mjx_vectorized_state.observations["joint_position"][0][j*2*n:(j+1)*2*n:2])
                joint_angles_oop_t.append(mjx_vectorized_state.observations["joint_position"][0][j*2*n+1:(j+1)*2*n+1:2])
                j += 1

        joint_angles_ip.append(joint_angles_ip_t)
        joint_angles_oop.append(joint_angles_oop_t)

        action = vectorized_nn_model_apply(policy_params_solution_shaped, sensory_input_pad)
        action_selection = select_actuator_output(action, arm_setup, arm_setup_damage)
        
        mjx_vectorized_state = mjx_vectorized_step(state=mjx_vectorized_state, action=action_selection)
        
        mjx_frames_damage.append(
                post_render(
                    mjx_vectorized_env.render(state=mjx_vectorized_state),
                    mjx_vectorized_env.environment_configuration
                    )
                )
        
    imgio_kargs = {
        'fps': fps, 'quality': 10, 'macro_block_size': None, 'codec': 'h264',
        'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']
        }
    writer = imageio.get_writer(VIDEO_DIR + run_name + " DAMAGE.mp4", **imgio_kargs)
    for frame in mjx_frames_damage:
        writer.append_data(frame)
    writer.close()

    wandb.log({"Video damaged morphology": wandb.Video(VIDEO_DIR + run_name + " DAMAGE.mp4",
                                                       caption=run_name+" DAMAGE", fps=fps, format='mp4')})
    
    fig, axes = plot_ip_oop_joint_angles(joint_angles_ip, joint_angles_oop)
    wandb.log({"Joint Angles damaged morophology": wandb.Image(fig)})

else:
    print('no damage simulation has been run')


wandb.finish()