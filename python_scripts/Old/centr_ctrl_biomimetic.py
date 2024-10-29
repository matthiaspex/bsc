import sys
import os

print(f"sys.executable: {sys.executable}")

import numpy as np
import jax
from jax import numpy as jnp
from evosax import OpenES, ParameterReshaper, NetworkMapper, FitnessShaper
import time
import wandb

# custom modules import --> from bsc_utils directory
from bsc_utils.visualization import visualize_mjcf, plot_ip_oop_joint_angles, save_video_from_raw_frames 
from bsc_utils.controller.base import ExplicitMLP
from bsc_utils.BrittleStarEnv import EnvContainer
from bsc_utils.damage import check_damage
from bsc_utils.simulate.base import rollout_old, generate_video_joint_angle_raw
from bsc_utils.miscellaneous import check_GPU_access, load_config_from_yaml, store_config_and_policy_params, get_run_name_from_config
from bsc_utils.evolution import efficiency_from_reward_cost, fitness_from_stacked_data




np.set_printoptions(precision=3, suppress=False, linewidth=100)
jnp.set_printoptions(precision=3, suppress=False, linewidth=100)

###############################################
# READOUT ALL ARGUMENTS FROM YAML/SYS VARIABLES
###############################################

VIDEO_DIR = os.environ["VIDEO_DIR"]
POLICY_PARAMS_DIR = os.environ["POLICY_PARAMS_DIR"]
CONFIG_FILE = os.environ["CONFIG_FILE"]


config = load_config_from_yaml(CONFIG_FILE)
rng = jax.random.PRNGKey(config["experiment"]["seed"])
run_name = get_run_name_from_config(config)

print(f"run_name: {run_name}")


check_GPU_access(interface = config["experiment"]["interface"])

env_container = EnvContainer(config)
env_container.generate_env() # env_container now has attributes for the environment, morphology_specification, arena_configuration and environment_configuration
env_container.generate_env_damaged() # env_container now also has attributes for env_damaged and morphology_specification_damaged

total_num_control_timesteps = env_container.environment_configuration.total_num_control_steps


##########################################
# instantiating the vectorized environment
##########################################
NUM_MJX_ENVIRONMENTS = config["evolution"]["es_popsize"] # best to get close to number of CUDA cores available


rng, mjx_vectorized_env_rng = jax.random.split(rng, 2)
mjx_vectorized_env_rng = jnp.array(jax.random.split(mjx_vectorized_env_rng, NUM_MJX_ENVIRONMENTS))

# vectorizing the functionalities
mjx_vectorized_step = jax.jit(jax.vmap(env_container.env.step))
mjx_vectorized_reset = jax.jit(jax.vmap(env_container.env.reset))

# reset the state of all parallel mjx environments
mjx_vectorized_state = mjx_vectorized_reset(mjx_vectorized_env_rng)

sensors = [key for key in mjx_vectorized_state.observations.keys()]
sensors_with_dim = {s: d.shape for s, d in mjx_vectorized_state.observations.items()}
sensor_selection_dim = sum([mjx_vectorized_state.observations[sensor].shape[-1] for sensor in config["environment"]["sensor_selection"]])

print(f"""
      sensors: {sensors}
      sensors with dimensions {sensors_with_dim}
      sensor selection: {config["environment"]["sensor_selection"]}
      sensor selection dimension: {sensor_selection_dim}
""")


sensory_input_nn_test = jnp.concatenate(
    [mjx_vectorized_state.observations[label] for label in config["environment"]["sensor_selection"]],
    axis = 1
)
print(f"""
      sensory_input_nn_test.shape: {sensory_input_nn_test.shape}
""")

nn_input_dim = sensor_selection_dim
nn_output_dim = len(env_container.env.actuators)
print(f"""
      Dimension of sensory space: {nn_input_dim}
      Dimension of actuator space: {nn_output_dim}
""")

###################################
# Instantiating the neural network
###################################

features = config["controller"]["hidden_layers"] + [nn_output_dim]
print(f"""
      features = {features}
      hidden_layers = {config["controller"]["hidden_layers"]}
      type(hidden_layers) = {type(config["controller"]["hidden_layers"])}
      """)
nn_model = ExplicitMLP(features = tuple(features), joint_control = config["morphology"]["joint_control"])


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


#####################################
# Instantiating the search strategy #
#####################################
rng, rng_ask, rng_init = jax.random.split(rng, 3)
strategy  = OpenES(popsize = config["evolution"]["es_popsize"], num_dims = num_params)
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
    project = config["wandb"]["project"],
    group = config["wandb"]["group"],
    name = run_name,
    tags = config["wandb"]["tags"],
    notes = config["wandb"]["notes"],
    config = config
    )

# Run ask-eval-tell loop
start_time = time.time()
for gen in range(config["evolution"]["num_generations"]):
    # track runtimes
    if gen%10 == 0:
        print('generation: ', gen, '\ntime since start: ', time.time()-start_time)
        
    
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    
    policy_params, es_state = strategy.ask(rng_gen, es_state, es_params)
    policy_params_shaped = param_reshaper.reshape(policy_params) # --> stacked pytree


    final_state, steps_stacked_data, rng = rollout_old(mjx_vectorized_env = env_container.env,
                                               nn_model = nn_model,
                                               policy_params_shaped = policy_params_shaped,
                                               total_num_control_timesteps = total_num_control_timesteps,
                                               sensor_selection = config["environment"]["sensor_selection"],
                                               rng = rng,
                                               NUM_MJX_ENVIRONMENTS = NUM_MJX_ENVIRONMENTS,
                                               cost_expr = config["evolution"]["cost_expr"],
                                               penal_expr = config["evolution"]["penal_expr"]
                                               )
    
    rewards, costs, penal = steps_stacked_data

    # rewards shape is: e.g. 300 parallel rewards per control step, an additional stack every control step --> sum over the control steps along axis = 0
    total_reward = jnp.sum(rewards, axis = 0)
    total_cost = jnp.sum(costs, axis = 0)
    total_penal = jnp.sum(penal, axis = 0)
    efficiency = efficiency_from_reward_cost(total_reward, total_cost, config["evolution"]["efficiency_expr"])
    fitness = fitness_from_stacked_data(stacked_data=steps_stacked_data, efficiency_expr=config["evolution"]["efficiency_expr"])

    # fitness should be an array with population size as len (e.g. 100)
    fit_re = fit_shaper.apply(policy_params, fitness)

   
    # log metrics to wandb
    wandb.log({"mean reward": jnp.mean(total_reward),
               "max reward":jnp.max(total_reward),
               "mean cost": jnp.mean(total_cost),
               "min cost": jnp.min(total_cost),
               "mean penalty": jnp.mean(total_penal),
               "min penalty": jnp.min(total_penal),
               "mean efficiency": jnp.mean(efficiency),
               "max efficiency": jnp.max(efficiency),
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



store_config_and_policy_params(file_name=POLICY_PARAMS_DIR+run_name, cfg=config, policy_params=policy_params_to_render)


#####################################
# Video and angle plots visualisation
#####################################
policy_params_to_render = jnp.array(policy_params_to_render)
fps = int(1/env_container.environment_configuration.control_timestep)
file_path_video = VIDEO_DIR + run_name + ".mp4"

rng, rng_render = jax.random.split(rng, 2)
frames, joint_angles_ip, joint_angles_oop = generate_video_joint_angle_raw(
                                                                    policy_params_to_render=policy_params_to_render,
                                                                    param_reshaper=param_reshaper,
                                                                    rng=rng_render,
                                                                    mjx_vectorized_env=env_container.env,
                                                                    sensor_selection=config["environment"]["sensor_selection"],
                                                                    arm_setup=config["morphology"]["arm_setup"],
                                                                    nn_model=nn_model
                                                                    )

save_video_from_raw_frames(
    frames=frames,
    fps=fps,
    file_path=file_path_video
    )
wandb.log({"Video trained model": wandb.Video(file_path_video, caption=run_name, fps=fps, format='mp4')})


fig, axes = plot_ip_oop_joint_angles(joint_angles_ip, joint_angles_oop)
wandb.log({"Joint Angles trained model": wandb.Image(fig)})


#########################################################
# Damage: Applying trained policy to a damaged morphology
#########################################################

if config["damage"]["damage"] == True:
    check_damage(arm_setup = config["morphology"]["arm_setup"], arm_setup_damage = config["damage"]["arm_setup_damage"])

    rng, rng_render = jax.random.split(rng, 2)
    frames, joint_angles_ip, joint_angles_oop = generate_video_joint_angle_raw(
                                                                        policy_params_to_render=policy_params_to_render,
                                                                        param_reshaper=param_reshaper,
                                                                        rng=rng_render,
                                                                        mjx_vectorized_env=env_container.env_damage,
                                                                        sensor_selection=config["environment"]["sensor_selection"],
                                                                        arm_setup=config["morphology"]["arm_setup"],
                                                                        nn_model=nn_model,
                                                                        damage = config["damage"]["damage"],
                                                                        arm_setup_damage = config["damage"]["arm_setup_damage"]
                                                                        )

    file_path_video_damage = VIDEO_DIR + run_name + " DAMAGE.mp4"
    save_video_from_raw_frames(
        frames=frames,
        fps=fps,
        file_path=file_path_video_damage
        )

    wandb.log({"Video damaged morphology": wandb.Video(file_path_video_damage, caption=run_name+" DAMAGE", fps=fps, format='mp4')})
    
    fig, axes = plot_ip_oop_joint_angles(joint_angles_ip, joint_angles_oop)
    wandb.log({"Joint Angles damaged morophology": wandb.Image(fig)})

else:
    print('no damage simulation has been run')





wandb.finish()
env_container.clear_envs() # closes the environments and resets internal environment values