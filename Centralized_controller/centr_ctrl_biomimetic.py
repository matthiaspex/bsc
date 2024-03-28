import argparse

# import yaml

# with open("config\\test.yaml") as f:
#     cfg = yaml.load(f, Loader=yaml.FullLoader)



# load arguments
parser = argparse.ArgumentParser(description='Centralized controller')

# main arguments
parser.add_argument('interface', type=str, help='either VSC or HPC, activates certain parts of code')
parser.add_argument('arm_setup', type=str, help='delimited string containing: number of segments for each of 5 arms')
parser.add_argument('reward_type', type=str, choices = ['distance', 'target', 'light'], help='choose from distance, target or light')

parser.add_argument('es_popsize', type=int, help='the number of parallel policy params to evaluate every generation')
parser.add_argument('num_generations', type=int, help='the number generations for es to run')

parser.add_argument('sensor_selection', type=str, help='delimited string containing: selected sensors')
parser.add_argument('layers', type=str, help='delimited string containing: neural network dense layers architecture')
parser.add_argument('damage', type=str, help='whether a damage simulation is run')

parser.add_argument('project', type=str, help='project title wandb')
parser.add_argument('notes', type=str, help='notes to pass on to wandb')

parser.add_argument('json_path', type = str, help='location to store policy params of trained models as .json files')
parser.add_argument('video_path', type = str, help='location to store videos of trained models when running in VSC')

# optional arguments with default values
parser.add_argument('--joint_control', choices = ['position', 'torque'], default = 'position', help='whether nn controls positions or torques of joints')
parser.add_argument('--sand_ground_color', type=str, default = 'False', help='whether a damage simulation is run')
parser.add_argument('--num_physics_steps_per_control_step', type=int, default = 10)
parser.add_argument('--simulation_time', type=int, default = 5, help='number of seconds before an episode finishes')
parser.add_argument('--joint_randomization_noise_scale', type=float, default = 0.0, help='whether joints are in a random position when initialized')
parser.add_argument('--light_perlin_noise_scale', type=int, default = 4, help='if 0: no light gradients; choose integer factor of 200')
parser.add_argument('--target_distance', type=float, default = 3.0, help='radius where target is located on')
parser.add_argument('--arm_setup_damage', type=str, help='delimited string containing damaged configuration: only used if damage=True')

args = parser.parse_args()

# readout the main arguments
interface = args.interface
arm_setup = [int(item) for item in args.arm_setup.split(',')]
reward_type = args.reward_type
es_popsize = args.es_popsize
num_generations = args.num_generations
sensor_selection = tuple([str(item) for item in args.sensor_selection.split(',')])
layers = [int(item) for item in args.layers.split(',')]
damage = args.damage=='True'
project = args.project
notes = args.notes
json_path = args.json_path
video_path = args.video_path

# readout the optional arguments or provide default value if not yet provided
joint_control = args.joint_control
sand_ground_color = args.sand_ground_color=='True'
num_physics_steps_per_control_step = args.num_physics_steps_per_control_step
simulation_time = args.simulation_time
joint_randomization_noise_scale = args.joint_randomization_noise_scale
light_perlin_noise_scale = args.light_perlin_noise_scale
target_distance = args.target_distance

if args.arm_setup_damage:
    arm_setup_damage = [int(item) for item in args.arm_setup_damage.split(',')]
else:
    arm_setup_damage = arm_setup

import os
if interface == 'HPC':
    TMP_DIR = os.environ["TMP_DIR"]
    print("TMP_DIR path: ", TMP_DIR)
    video_path = TMP_DIR



import sys
print("sys.executable: ", sys.executable)
if interface == 'VSC':
    print('sys path insert enabled for VSC')
    sys.path.insert(0,'C:\\Users\\Matthias\\OneDrive - UGent\\Documents\\DOCUMENTEN\\3. Thesis\\BSC\\')


print(f"""
      damage = {damage}
      arm_setup = {arm_setup}
      sensor_selection = {sensor_selection}
      layers = {layers}
      layers.append(5) = {layers + [5]}
      target_distance = {target_distance}
      notes = {notes}
      sand_ground_color = {sand_ground_color}
      """)

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

import json

import moviepy
import imageio
import plotly
import cv2
import mediapy as media

from bsc_utils.miscallaneous import check_GPU_access

check_GPU_access(interface = interface)





    

np.set_printoptions(precision=3, suppress=False, linewidth=100)
jnp.set_printoptions(precision=3, suppress=False, linewidth=100)

rng = jax.random.PRNGKey(0) # make an rng right away and every split throughout the document should make a new rng
# this new rng should only be used for the sole purpose of splitting in the future

# custom modules import --> from bsc_utils directory
from bsc_utils.visualization import visualize_mjcf, show_video, create_video, post_render 
from bsc_utils.controller import ExplicitMLP
from bsc_utils.BrittleStarEnv import create_morphology, create_arena, create_environment
from bsc_utils.damage import check_damage, pad_sensory_input, select_actuator_output


# some specific inports from biorobot
from biorobot.brittle_star.mjcf.morphology.specification.default import default_brittle_star_morphology_specification
from biorobot.brittle_star.environment.light_escape.shared import BrittleStarLightEscapeEnvironmentConfiguration
from biorobot.brittle_star.environment.directed_locomotion.shared import \
    BrittleStarDirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.environment.undirected_locomotion.shared import \
    BrittleStarUndirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.mjcf.arena.aquarium import AquariumArenaConfiguration, MJCFAquariumArena





def rollout(policy_params_shaped, total_num_control_timesteps, sensor_selection, rng, NUM_MJX_ENVIRONMENTS):
    """
    Do a single episode rollout
    Inputs:
    - Reshaped policy params to match neural network
    - Total_num_control_timesteps
    - sensor selection
    - rng
    - Total number of parallel MJX environments
    Outpus:
    - Final MJX environment state (vectorized)
    - Rewards: sum of all rewards of every timestep
    - rng
    """
    # add and return 'rng' param?
    rng, mjx_vectorized_env_rng = jax.random.split(rng, 2)
    mjx_vectorized_env_rng = jnp.array(jax.random.split(mjx_vectorized_env_rng, NUM_MJX_ENVIRONMENTS))
    mjx_vectorized_state_reset = mjx_vectorized_reset(rng=mjx_vectorized_env_rng)
    
    carry_init = [mjx_vectorized_state_reset, policy_params_shaped]

    def step(carry, xs):
        """
        Carry: any information required in the carryover:
        - vectorized state array for the mujoco steps
        - shaped pytree with policy params for the model.apply
        -----
        xs is not required for this purpose, but put in the arguments as a placeholder for jax.lax.scan to function.
        -----
        output:
        - Carry: state to be updated, and the same policy_params
        - stack: rewarsd and work costs are tracked for every step
        """    

        _mjx_vectorized_state, _policy_params_shaped = carry

        _sensory_input_nn = jnp.concatenate(
            [_mjx_vectorized_state.observations[label] for label in sensor_selection],
            axis = 1
        )
            
        # # below: if you want to select all the observations
        # _sensory_input_nn = jnp.concatenate(
        #     jax.tree_util.tree_map(_mjx_vectorized_state.observations.__getitem__, list(_mjx_vectorized_state.observations.keys())),
        #     axis = 1
        # )
        
        
        _action = vectorized_model_apply(_policy_params_shaped, _sensory_input_nn)
        
        _mjx_vectorized_state_updated = mjx_vectorized_step(state=_mjx_vectorized_state, action=_action)
        
        reward_step = _mjx_vectorized_state_updated.reward


        joint_velocities = _mjx_vectorized_state.observations['joint_velocity']
        torques = _mjx_vectorized_state.observations['joint_actuator_force']
        power_step = jnp.sum(jnp.abs(joint_velocities)*jnp.abs(torques), axis = 1)
        # power_step = jnp.sum(jnp.abs(joint_velocities)*jnp.abs(torques) + jnp.abs(torques), axis = 1)
        # power_step = jnp.sum(jnp.abs(torques), axis = 1)

        
        # above sum sums over all the segments of 1 single individual in the generation
        # summing this power over every time step results in the total work done (done outside of the rollout function)

        carry = [_mjx_vectorized_state_updated, _policy_params_shaped]
        return carry, [reward_step, power_step]

    final_carry, stack = jax.lax.scan(
        step,
        carry_init,
        None,
        total_num_control_timesteps
    )

    rewards, works = stack
    
    mjx_vectorized_state_final = final_carry[0]
    return mjx_vectorized_state_final, rewards, works, rng

rollout = jax.jit(rollout, static_argnames=("total_num_control_timesteps", "NUM_MJX_ENVIRONMENTS", "sensor_selection"))



# arm_setup = [5,0,5,0,0] # 2 arms with 5 segments
dofs = 2*sum(arm_setup)
print("Number of actuator outputs: dofs = ", dofs)



# reward_type = "distance" # choose "distance", "target", "light"
# joint_control = "position" # choose "position" or "torque"
# sand_ground_color = False # choose True or False, False allows for seeing movement with respect to grid lines.
# num_physics_steps_per_control_step=10 
# simulation_time=5 # [seconds]
# joint_randomization_noise_scale=0.0
# light_perlin_noise_scale=4 # default: take 0 for no noise, take max of 200, take integer factor of 200
# # If this value is > 0 then we will add randomly sampled noise to the initial joint positions and velocities


# specifying morphology
morphology_specification = default_brittle_star_morphology_specification(
        num_arms=len(arm_setup), num_segments_per_arm=arm_setup, use_p_control=(joint_control == 'position'), use_torque_control=(joint_control == 'torque')
        )
morphology = create_morphology(morphology_specification=morphology_specification)
visualize_mjcf(mjcf=morphology)


# specifying arena
arena_configuration = AquariumArenaConfiguration(
        size=(10, 5), sand_ground_color=sand_ground_color, attach_target=(reward_type=="target"), wall_height=1.5, wall_thickness=0.1
        )
arena = create_arena(arena_configuration=arena_configuration)
visualize_mjcf(mjcf=arena)


# specifying environment: CHOOSE 1
assert reward_type in (["distance","target","light"]), "reward_type must be one of 'distance', 'target', 'light'"

if reward_type == "distance":
    environment_configuration = BrittleStarUndirectedLocomotionEnvironmentConfiguration(
        # If this value is > 0 then we will add randomly sampled noise to the initial joint positions and velocities 
        joint_randomization_noise_scale=joint_randomization_noise_scale,
        render_mode="rgb_array",  # Visualization mode 
        simulation_time=simulation_time,  # Number of seconds per episode 
        num_physics_steps_per_control_step=num_physics_steps_per_control_step,  # Number of physics substeps to do per control step 
        time_scale=2,    # Integer factor by which to multiply the original physics timestep of 0.002,
        camera_ids=[0, 1],   # Which camera's to render (all the brittle star environments contain 2 cameras: 1 top-down camera and one close-up camera that follows the brittle star),
        render_size=(480, 640)  # Resolution to render with ((height, width) in pixels)
        )

elif reward_type == "target":
    environment_configuration = BrittleStarDirectedLocomotionEnvironmentConfiguration(
        # Distance to put our target at (targets are spawned on a circle around the starting location with this given radius).
        target_distance=target_distance,
        joint_randomization_noise_scale=joint_randomization_noise_scale,
        render_mode="rgb_array",
        simulation_time=simulation_time,
        num_physics_steps_per_control_step=num_physics_steps_per_control_step,
        time_scale=2,
        camera_ids=[0, 1],
        render_size=(480, 640)
        )

elif reward_type == "light":
    environment_configuration = BrittleStarLightEscapeEnvironmentConfiguration(
        joint_randomization_noise_scale=joint_randomization_noise_scale,
        light_perlin_noise_scale=light_perlin_noise_scale,
        # If this value is > 0, we will add perlin noise to the generated light map. Otherwise, the light map is a simple linear gradient.
        # Please only provide integer factors of 200.
        render_mode="rgb_array",
        simulation_time=simulation_time,
        num_physics_steps_per_control_step=num_physics_steps_per_control_step,
        time_scale=2,
        camera_ids=[0, 1],
        render_size=(480, 640)
        )

BACKEND = "MJX"

# useful environment configuration information
print(f"[simulation_time] The total amount of time (in seconds) that one simulation episode takes: {environment_configuration.simulation_time}")
print(f"[physics_timestep] The amount of time (in seconds) that one 'physics step' advances the physics: {environment_configuration.physics_timestep}")
print(f"[control_timestep] The amount of time (in seconds) that one 'control step' advances the physics: {environment_configuration.control_timestep}")
print(f"[total_num_physics_steps] The total amount of physics steps that happen during one simulation episode: {environment_configuration.total_num_physics_steps}")
print(f"[total_num_control_steps] The total amount of control steps that happen during one simulation episode: {environment_configuration.total_num_control_steps}")


total_num_control_timesteps = environment_configuration.total_num_control_steps


##########################################
# instantiating the vectorized environment
##########################################
# es_popsize = 400
NUM_MJX_ENVIRONMENTS = es_popsize # best to get close to number of CUDA cores available
# num_generations = 50

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
mjx_vectorized_state = mjx_vectorized_reset(rng=mjx_vectorized_env_rng)


sensors = [key for key in mjx_vectorized_state.observations.keys()]
sensors_with_dim = {s: d.shape for s, d in mjx_vectorized_state.observations.items()}
# sensor_selection = ('joint_position', 'joint_actuator_force', 'segment_contact')
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

# features = [128,128, nn_output_dim]
features = layers + [nn_output_dim]
print(f"""
      features = {features}
      layers = {layers}
      type(layers) = {type(layers)}
      """)
model = ExplicitMLP(features = features, joint_control = joint_control)

# initialising the parameters of the model
rng, rng_input, rng_init = jax.random.split(rng, 3)
x = jnp.zeros(nn_input_dim) # just required for the initialisation: only length matters, not input values
print("shape of the input layer nn: ", x.shape)
policy_params_init = model.init(rng_init, x)

# vectorize the model.apply function
vectorized_model_apply = jax.jit(jax.vmap(model.apply))


# params is a PyTree --> see jax documentation
# print(params)
print(f"""nn features: {features}
      policy_params_init pytree keys: {policy_params_init['params'].keys()}
      policy params pytree description: {jax.tree_util.tree_map(lambda x: x.shape, policy_params_init)}
""")

# single forward pass through the model
test_output = model.apply(policy_params_init,jax.random.uniform(rng_input, (nn_input_dim,)))
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

# Check some params:
print(f"""
-----Checking the setup for optimisation-----
Total number of control timesteps per episode: {total_num_control_timesteps}
Total simulation time [seconds]: {simulation_time}s
Selected sensors: {sensor_selection}

Neural network architecture: {[nn_input_dim] + features} with:
\tSensory input layer dimension: {nn_input_dim}:
\tActuator output dimension: {features[-1]}
\tNumber of hidden layers / nodes per hidden layer: {features[:-1]}
\tNumber of policy params: {num_params}

Arm setup: {arm_setup}
Joint control: {joint_control}
Reward_type: {reward_type}

Number of generations for updating policy parameters: {num_generations}
Population size per generation of policy parameters: {es_popsize}
Number of parallel MJX environments: {NUM_MJX_ENVIRONMENTS}
""")

# reset the search strategy state
rng, rng_init = jax.random.split(rng, 2)
es_state = strategy.initialize(rng_init, es_params)

policy_params_to_render = []

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project = project,
    # set run name
    name = f"{reward_type} {joint_control} arms {arm_setup} popsize {es_popsize} {notes}",
    # put notes that are queried by the user
    notes = notes,

    # track hyperparameters and run metadata
    config={
    "nn_architecture": [nn_input_dim] + features,
    "arm_setup": arm_setup,
    "num_params": num_params,
    "activation_function": "Tanh",
    "num_generations": num_generations,
    "es_popsize": es_popsize,
    }
)

# Run ask-eval-tell loop - NOTE: By default minimization!
start_time = time.time()
for gen in range(num_generations):
    # track runtimes
    if gen%10 == 0:
        print('generation: ', gen, '\ntime since start: ', time.time()-start_time)
        
    
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    
    policy_params, es_state = strategy.ask(rng_gen, es_state, es_params)
    policy_params_shaped = param_reshaper.reshape(policy_params) # --> stacked pytree


    final_state, rewards, works, rng = rollout(policy_params_shaped = policy_params_shaped,
                                                   total_num_control_timesteps = total_num_control_timesteps,
                                                   sensor_selection = sensor_selection,
                                                   rng = rng,
                                                   NUM_MJX_ENVIRONMENTS = NUM_MJX_ENVIRONMENTS
                                              )

    # rewards shape is: 300 parallel rewards per control step, an additional stack every control step --> sum over the control steps along axis = 0
    total_reward = jnp.sum(rewards, axis = 0)
    total_work = jnp.sum(works, axis = 0)
    total_work_rescaled = total_work * jnp.mean(total_reward)/jnp.mean(total_work)
    # if gen < 20:
    #     fitness = total_reward
    # else:
    #     # fitness = total_reward + total_reward/total_work_rescaled
    #     fitness = total_reward/total_work_rescaled
    
    # fitness = total_reward/total_work_rescaled

    # fitness = total_reward + total_reward/total_work_rescaled

    fitness = total_reward/total_work

        
    # fitness should be an array with population size as len (e.g. 100)
    fit_re = fit_shaper.apply(policy_params, fitness)

   
    # log metrics to wandb
    wandb.log({"mean reward": jnp.mean(total_reward),
               "max reward":jnp.max(total_reward),
               "mean work": jnp.mean(total_work),
               "work rescaled": jnp.mean(total_work_rescaled),
               "mean fitness (reward/work)": jnp.mean(fitness)
              })
    
    # # Select certain training generations to render videos
    # if gen in [2, 5, 10, 20, 40, 80]:
    #     policy_params_to_render.append(es_state.best_member)
    
    es_state = strategy.tell(policy_params, fit_re, es_state, es_params)
    


# # Get best overall population member & its fitness
# es_state.best_member, es_state.best_fitness
policy_params_to_render.append(policy_params[jnp.argmax(fitness)])
print('Policy training finished!')

# # render the es method 'best member', the member with max reward and member with max fitness
# policy_params_to_render = []
# policy_params_to_render.append(es_state.best_member)
# policy_params_to_render.append(policy_params[jnp.argmax(fitness)])
# policy_params_to_render.append(policy_params[jnp.argmax(total_reward)])
# print(jnp.argmax(fitness))
# print(jnp.argmax(total_reward))

# print(f"""best member: {es_state.best_member}
#       best member fitness: {es_state.best_fitness}
#       """)
# print(policy_params[jnp.argmax(fitness)])
# print(policy_params[jnp.argmax(total_reward)])

# print(f"""
#       difference 'best member' en best fitness: {es_state.best_member-policy_params[jnp.argmax(fitness)]}
#       difference 'best member' en best total reward: {es_state.best_member-policy_params[jnp.argmax(total_reward)]}
#       difference best fitness and best total reward: {policy_params[jnp.argmax(total_reward)]-policy_params[jnp.argmax(fitness)]}
# """)


policy_dump = policy_params_to_render.json()
file = json_path + "policy_params_test.json"
with open(file, "w") as file:
    json.dump(policy_dump, file)
file.close()


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

    action = vectorized_model_apply(policy_params_solution_shaped, sensory_input)
    
    mjx_vectorized_state = mjx_vectorized_step(state=mjx_vectorized_state, action=action)

    
    mjx_frames.append(
            post_render(
                mjx_vectorized_env.render(state=mjx_vectorized_state),
                mjx_vectorized_env.environment_configuration
                )
            )

# show_video(images=mjx_frames, fps = fps)
# create_video(frames=mjx_frames, framerate=fps, out_path=f"Videos\{reward_type} {joint_control} arms {arm_setup} popsize {es_popsize} {notes}.mp4")
imgio_kargs = {
    'fps': fps, 'quality': 10, 'macro_block_size': None, 'codec': 'h264',
    'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']
    }
writer = imageio.get_writer(video_path + f"{reward_type} {joint_control} arms {arm_setup} popsize {es_popsize} {notes}.mp4", **imgio_kargs)
for frame in mjx_frames:
    writer.append_data(frame)
writer.close()

wandb.log({"Video trained model": wandb.Video(video_path + f"{reward_type} {joint_control} arms {arm_setup} popsize {es_popsize} {notes}.mp4",
                                              caption="5 arms torques cost trial run", fps=fps, format='mp4')})




joint_angles_ip = np.array(joint_angles_ip)
joint_angles_oop = np.array(joint_angles_oop)
print(f"shape explanation: (t, number of arms, number of segment) = {joint_angles_ip.shape}")
t, num_arms, num_segments = joint_angles_ip.shape


plt.rcParams['figure.figsize'] = (5*num_arms,5*num_segments)



fig, axes = plt.subplots(num_segments,num_arms)


for i in range(num_segments):
    for j in range(num_arms):

        axes[i][j].plot(joint_angles_ip[:,j,i], joint_angles_oop[:,j,i])
        axes[i][j].set_title(f"Joint angles for segment {i} in arm {j}")
        axes[i][j].set_xlabel("In plane joint angle [rad]")
        axes[i][j].set_ylabel("Out of plane joint angle [rad]")
    
# plt.savefig('distance position arms 50500 popsize 400 (distance 3.5).jpg') # --> rather put it in wandb as a plot
wandb.log({"Joint Angles trained model": wandb.Image(fig)})
plt.show()




#########################################################
# Damage: Applying trained policy to a damaged morphology
#########################################################


# arm_setup_damage = [5,0,0,0,0]

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
        for n in arm_setup:
            if n != 0:
                joint_angles_ip_t.append(mjx_vectorized_state.observations["joint_position"][0][j*2*n:(j+1)*2*n:2])
                joint_angles_oop_t.append(mjx_vectorized_state.observations["joint_position"][0][j*2*n+1:(j+1)*2*n+1:2])
                j += 1

        joint_angles_ip.append(joint_angles_ip_t)
        joint_angles_oop.append(joint_angles_oop_t)

        action = vectorized_model_apply(policy_params_solution_shaped, sensory_input_pad)
        action_selection = select_actuator_output(action, arm_setup, arm_setup_damage)
        
        mjx_vectorized_state = mjx_vectorized_step(state=mjx_vectorized_state, action=action_selection)

        
        mjx_frames_damage.append(
                post_render(
                    mjx_vectorized_env.render(state=mjx_vectorized_state),
                    mjx_vectorized_env.environment_configuration
                    )
                )

    # show_video(images=mjx_frames_damage, fps=fps)
    # create_video(frames=mjx_frames_damage, framerate=fps, out_path=f"Videos\{reward_type} {joint_control} arms {arm_setup} popsize {es_popsize} {notes} DAMAGE.mp4")
    imgio_kargs = {
        'fps': fps, 'quality': 10, 'macro_block_size': None, 'codec': 'h264',
        'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']
        }
    writer = imageio.get_writer(video_path + f"{reward_type} {joint_control} arms {arm_setup} popsize {es_popsize} {notes} DAMAGE.mp4", **imgio_kargs)
    for frame in mjx_frames_damage:
        writer.append_data(frame)
    writer.close()

    wandb.log({"Video damaged morphology": wandb.Video(video_path + f"{reward_type} {joint_control} arms {arm_setup} popsize {es_popsize} {notes} DAMAGE.mp4",
                                                       caption="5 arms torques cost trial run", fps=fps, format='mp4')})

else:
    print('no damage simulation has been run')





wandb.finish()
