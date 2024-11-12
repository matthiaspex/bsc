"""
This file is compatible with any config (yaml) file that takes Hebbian True or not (so also static trainings are possible in this file)
"""

import sys
import os

print(f"sys.executable: {sys.executable}")

import jax
from jax import numpy as jnp
from evosax import OpenES, FitnessShaper
import time
import wandb

from bsc_utils.controller.base import ExplicitMLP, NNController
from bsc_utils.controller.hebbian import HebbianController
from bsc_utils.simulate.train import Trainer
from bsc_utils.simulate.analyze import Simulator
from bsc_utils.simulate.base import rollout
from bsc_utils.miscellaneous import check_GPU_access, load_config_from_yaml, \
    store_config_and_policy_params, get_run_name_from_config, complete_sensor_selection, \
    get_target_positions
from bsc_utils.evolution import efficiency_from_reward_cost, fitness_from_stacked_data

jnp.set_printoptions(precision=3, suppress=False, linewidth=100)

###############################################
# READOUT ALL ARGUMENTS FROM YAML/SYS VARIABLES
###############################################

VIDEO_DIR = os.environ["VIDEO_DIR"] # where to store generated videos
IMAGE_DIR = os.environ["IMAGE_DIR"] # where to store generated videos
POLICY_PARAMS_DIR = os.environ["POLICY_PARAMS_DIR"] # where to store trained policy params
CONFIG_FILE = os.environ["CONFIG_FILE"]

config = load_config_from_yaml(CONFIG_FILE)


# in case of target/light reward: add those relevant inputs to the sensor_selection
config = complete_sensor_selection(config)

rng = jax.random.PRNGKey(config["experiment"]["seed"])
run_name = get_run_name_from_config(config)

print(f"run_name: {run_name}")

check_GPU_access(interface=config["experiment"]["interface"])

trainer = Trainer(config) # trainer inherits from EnvContainer
trainer.generate_env() # env_container now has attributes for the environment, morphology_specification, arena_configuration and environment_configuration
# trainer.generate_env_damaged() # env_container now also has attributes for env_damaged and morphology_specification_damaged
observation_space_dim, action_space_dim = trainer.get_observation_action_space_info()

print(f"""
      Observation space dim: {observation_space_dim}
      Action space dim: {action_space_dim}
""")


if config["controller"]["hebbian"] == True:
    controller = HebbianController(trainer) # inherits from NNController
else:
    controller = NNController(trainer)
controller.update_model(ExplicitMLP) # all information to build controller present in EnvContainer (hidden layers, action dim, obs dim)
controller.update_parameter_reshaper() # all information for the parameterreshaper is present in the design from controller
param_reshaper = controller.get_parameter_reshaper()
num_params = param_reshaper.total_params

print(f"""
      Hebbian = {config["controller"]["hebbian"]}
      number of policy parameters: {num_params}
      """)



NUM_MJX_ENVIRONMENTS = config["evolution"]["es_popsize"] # best to get close to number of CUDA cores available

###################################
# Instantiating the search strategy
###################################
rng, rng_ask, rng_init = jax.random.split(rng, 3)
strategy  = OpenES(popsize = config["evolution"]["es_popsize"], num_dims = num_params)
# still parameters that can be finetuned, like optimisation method, lrate, lrate decay, ...
es_params = strategy.default_params
# # replacing certain parameters:
# es_params = es_params.replace(init_min = -3, init_max = 3)
print(f"es_params: {es_params}\n")

fit_shaper = FitnessShaper(maximize=True,
                           centered_rank=config["evolution"]["centered_rank"]
                           )

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

# run one ask-eval-tell loop (iterate over generations)
start_time = time.time()
for gen in range(config["evolution"]["num_generations"]):
    start_time_gen = time.time()

    print(f"generations: {gen}")
    # track runtimes
    if gen%10 == 0:
        print('generation: ', gen, '\ntime since start: ', start_time_gen-start_time)

    rng, rng_gen, rng_eval = jax.random.split(rng,3)

    policy_params_evosax, es_state = strategy.ask(rng_gen, es_state, es_params) # can be with learning rule dim (hebbian) or not

    if config["environment"]["reward_type"] == "target":
        rng, rng_targets = jax.random.split(rng, 2)
        targets = get_target_positions(rng=rng_targets,
                                        distance=config["training"]["target"]["distance"],
                                        num_rowing=config["training"]["target"]["num_rowing"],
                                        num_reverse_rowing=config["training"]["target"]["num_reverse_rowing"],
                                        num_random_positions=config["training"]["target"]["num_random_positions"],
                                        parallel_dim=NUM_MJX_ENVIRONMENTS,
                                        parallel_constant=config["training"]["target"]["parallel_constant"])
        total_reward_list = []
        total_cost_list = []
        total_penal_list = []
        efficiency_list = []
        fitness_list = []
        for i in range(len(targets)):
            vectorized_env_state_final, steps_stacked_data, rng = rollout(rng=rng,
                                                policy_params_evosax=policy_params_evosax,
                                                env=trainer.env,
                                                controller=controller,
                                                parallel_dim=NUM_MJX_ENVIRONMENTS,
                                                targets=targets[i]
                                                )
            rewards, costs, penal = steps_stacked_data
            total_reward_list.append(jnp.sum(rewards, axis = 0))
            total_cost_list.append(jnp.sum(costs, axis = 0))
            total_penal_list.append(jnp.sum(penal, axis = 0))
            efficiency_list.append(efficiency_from_reward_cost(total_reward_list[-1], total_cost_list[-1], config["evolution"]["efficiency_expr"]))
            fitness_list.append(fitness_from_stacked_data(stacked_data=steps_stacked_data, efficiency_expr=config["evolution"]["efficiency_expr"]))
        
        # list of number of targets (e.g. 4) arrays of popsize (e.g. 6912) rewards. take min/max along the targets dimension (axis = 0)
        total_reward = jnp.min(jnp.array(total_reward_list), axis = 0)
        total_cost = jnp.max(jnp.array(total_cost_list), axis = 0)
        total_penal = jnp.max(jnp.array(total_penal_list), axis = 0)
        efficiency = jnp.min(jnp.array(efficiency_list), axis = 0)
        fitness = jnp.min(jnp.array(fitness_list), axis = 0)

    else:
        vectorized_env_state_final, steps_stacked_data, rng = rollout(rng=rng,
                                            policy_params_evosax=policy_params_evosax,
                                            env=trainer.env,
                                            controller=controller,
                                            parallel_dim=NUM_MJX_ENVIRONMENTS
                                            )
    
        rewards, costs, penal = steps_stacked_data
        
        # rewards shape is: e.g. 6912 parallel rewards per control step, an additional stack every control step --> sum over the control steps along axis = 0
        total_reward = jnp.sum(rewards, axis = 0)
        total_cost = jnp.sum(costs, axis = 0)
        total_penal = jnp.sum(penal, axis = 0)
        efficiency = efficiency_from_reward_cost(total_reward, total_cost, config["evolution"]["efficiency_expr"])
        fitness = fitness_from_stacked_data(stacked_data=steps_stacked_data, efficiency_expr=config["evolution"]["efficiency_expr"])


    # fitness should be an array with population size as len (e.g. 6912)
    fit_re = fit_shaper.apply(policy_params_evosax, fitness)


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
               "max fitness": jnp.max(fitness),
               "time": time.time()-start_time_gen
              })

    # # Select certain training generations to render videos
    # if gen in [2, 5, 10, 20, 40, 80]:
    #     policy_params_to_render.append(es_state.best_member)

    es_state = strategy.tell(policy_params_evosax, fit_re, es_state, es_params)
    # if gen%150 == 0 and gen != 0:
    #     policy_params_to_render.append(policy_params_evosax[jnp.argmax(fitness)])
    #     store_config_and_policy_params(file_name=POLICY_PARAMS_DIR+run_name+f" gen: {gen}", cfg=config, policy_params=policy_params_to_render)
        
# Get best overall population member
policy_params_to_render.append(policy_params_evosax[jnp.argmax(fitness)])
print('Policy training finished!')

store_config_and_policy_params(file_name=POLICY_PARAMS_DIR+run_name, cfg=config, policy_params=policy_params_to_render)

#####################################
# Video and angle plots visualisation
#####################################
policy_params_to_render = jnp.array(policy_params_to_render)

simulator = Simulator(config)
simulator.generate_env()
simulator.generate_env_damaged()

controller.update_policy_params(policy_params=policy_params_to_render)

rng, rng_targets_simulator = jax.random.split(rng, 2)

if config["environment"]["reward_type"] == "target":
    targets_simulator = get_target_positions(rng=rng_targets_simulator,
                                    distance=config["environment"]["target_distance"],
                                    num_rowing=0,
                                    num_reverse_rowing=0,
                                    num_random_positions=1,
                                    parallel_dim=policy_params_to_render.shape[0],
                                    parallel_constant=True)
    simulator.update_targets(targets_simulator[0])

simulator.update_nn_controller(controller)

# simulate undamaged:
print("simulation of single episode started: Undamaged")
rng, rng_episode = jax.random.split(rng, 2)
simulator.generate_episode_data_undamaged(rng_episode)
print("simulation of single episode finished: Undamaged")

reward = simulator.get_episode_reward()
cost  = simulator.get_episode_cost()
penalty = simulator.get_episode_penalty()
efficiency = simulator.get_episode_efficiency()
fitness = simulator.get_episode_fitness()
simulator.get_ip_oop_joint_angles_plot(file_path = IMAGE_DIR + run_name + ".png")
simulator.get_episode_video(file_path = VIDEO_DIR + run_name + ".mp4")
simulator.get_kernel_animation(file_path = VIDEO_DIR + run_name + " kernel" + ".mp4")
simulator.get_final_kernel_histogram(file_path=IMAGE_DIR + run_name + " histogram" + ".png",\
                                        xlabel="synapse weights",\
                                        title="Final weight distribution - Undamaged")


print(f"""
reward = {reward}
cost = {cost}
penalty = {penalty}
efficiency = {efficiency}
fitness = {fitness}
""")

if config["damage"]["damage"]:
    print("simulation of single episode started: Damaged")
    rng, rng_episode = jax.random.split(rng, 2)
    simulator.generate_episode_data_damaged(rng_episode)
    print("simulation of single episode finished: Damaged")

    reward_damage = simulator.get_episode_reward()
    cost_damage  = simulator.get_episode_cost()
    penalty_damage = simulator.get_episode_penalty()
    efficiency_damage = simulator.get_episode_efficiency()
    fitness_damage = simulator.get_episode_fitness()
    simulator.get_ip_oop_joint_angles_plot(file_path = IMAGE_DIR + run_name + " DAMAGE.png")
    simulator.get_episode_video(file_path = VIDEO_DIR + run_name + " DAMAGE.mp4")
    simulator.get_kernel_animation(file_path = VIDEO_DIR + run_name + " kernel" + " DAMAGE.mp4")
    simulator.get_final_kernel_histogram(file_path=IMAGE_DIR + run_name + " histogram" + "DAMAGE.png",\
                                            xlabel="synapse weights",\
                                            title="Final weight distribution - Damaged")

    print(f"""
    reward_damage = {reward_damage}
    cost_damage = {cost_damage}
    penalty_damage = {penalty_damage}
    efficiency_damage = {efficiency_damage}
    fitness_damage = {fitness_damage}
    """)

else:
    print('no damage simulation has been run')


if config["damage"]["damage"]:
    print(f"""
    reward = {reward} - reward_damage = {reward_damage}
    cost = {cost} - cost_damage = {cost_damage}
    penalty = {penalty} - penalty_damage = {penalty_damage}
    efficiency = {efficiency} - efficiency_damage = {efficiency_damage}
    fitness = {fitness} - fitness_damage = {fitness_damage}
    """)



fps = int(1/simulator.environment_configuration.control_timestep)
wandb.log({"Video trained model": wandb.Video(VIDEO_DIR + run_name + ".mp4", caption=run_name, fps=fps, format='mp4')})
wandb.log({"Joint Angles trained model": wandb.Image(IMAGE_DIR + run_name + ".png")})
wandb.log({"Kernel visualisation": wandb.Video(VIDEO_DIR + run_name + " kernel" + ".mp4", caption=run_name, fps=fps, format='mp4')})
wandb.log({"Kernel visualisation": wandb.Image(IMAGE_DIR + run_name + " histogram" + ".png", caption=run_name, fps=fps, format='mp4')})

wandb.log({"Video damaged morphology": wandb.Video(VIDEO_DIR + run_name + " DAMAGE.mp4", caption=run_name, fps=fps, format='mp4')})
wandb.log({"Joint Angles damaged morophology": wandb.Image(IMAGE_DIR + run_name + " DAMAGE.png")})
wandb.log({"Kernel visualisation damaged": wandb.Video(VIDEO_DIR + run_name + " kernel" + " DAMAGE.mp4", caption=run_name, fps=fps, format='mp4')})
wandb.log({"Kernel visualisation": wandb.Image(IMAGE_DIR + run_name + " histogram" + "DAMAGE.png", caption=run_name, fps=fps, format='mp4')})



wandb.finish()
trainer.clear_envs()
simulator.clear_envs()
print("Environments cleared")



