import chex
from typing import Sequence, Union
import numpy as np

import jax
from jax import numpy as jnp
from evosax import ParameterReshaper

from moojoco.environment.mjx_env import MJXEnv, MJXEnvState

from bsc_utils.controller.base import ExplicitMLP, NNController
from bsc_utils.controller.hebbian import HebbianController
from bsc_utils.visualization import post_render, change_alpha, move_camera
from bsc_utils.damage import pad_sensory_input, select_actuator_output, check_damage

def rollout(
        rng: chex.PRNGKey,
        policy_params_evosax: chex.Array,
        env: MJXEnv,
        controller: Union[HebbianController, NNController],
        parallel_dim: int
        ):
    """
    Do a single episode rollout
    Inputs:
    - rng: jax rng key
    - mjx_env: Mujoco Environment as generated in the EnvContainer class:
        Can be damaged or non-damaged
        Is still parallelised within the rollout function
    - controller: NNController (of HebbianController which inherits from NNController)
        Should contain policy parameters already
        Initialised with EnvContainer so already has config information
    - parallel_dim: Total number of parallel MJX environments
    Outputs:
    - Final MJX environment state (vectorized)
    - Rewards: sum of all rewards of every timestep
    - rng
    """
    config = controller.config
    vectorized_controller_apply = jax.jit(jax.vmap(controller.apply))
    vectorized_env_step = jax.jit(jax.vmap(env.step))
    vectorized_env_reset = jax.jit(jax.vmap(env.reset))
    rng, vectorized_env_rng = jax.random.split(rng, 2)
    vectorized_env_rng = jnp.array(jax.random.split(vectorized_env_rng, parallel_dim))
    if config["environment"]["reward_type"] == 'target':
        vectorized_env_state_reset = vectorized_env_reset(rng=vectorized_env_rng, target_position=config["environment"]["target_position"])
    else:
        vectorized_env_state_reset = vectorized_env_reset(rng=vectorized_env_rng)

    controller.update_policy_params(policy_params=policy_params_evosax)
    policy_params = controller.get_policy_params()
    if config["controller"]["hebbian"] == True:
        rng, rng_ss_reset = jax.random.split(rng, 2)
        controller.reset_synapse_strengths_unif(rng_ss_reset, parallel_dim=parallel_dim)
        controller.reset_neuron_activities(parallel_dim=parallel_dim)
        synapse_strengths_init = controller.get_synapse_strengths()
        neuron_activities_init = controller.get_neuron_activities()
    else:
        synapse_strengths_init = controller.get_policy_params() # can be shaped or in flat evosax format
        controller.reset_neuron_activities(parallel_dim=parallel_dim)
        neuron_activities_init = controller.get_neuron_activities()

    carry_init = [vectorized_env_state_reset, synapse_strengths_init, policy_params, neuron_activities_init]

    def step(carry, _):
        """
        Carry: any information required in the carryover:
        - vectorized state array for the mujoco steps
        - synapse_strengths_init: shaped pytree or flat array with initial synapse_strengths
        - policy_params (remains static throughout steps, is not updated)
        - initial neuron activities (mainly necessary for Hebbian Learning)
        -----
        "_" placeholder for xs: is not required for this purpose, but put in the arguments as a placeholder for jax.lax.scan to function.
        -----
        output:
        - Carry: updated state, synapse strengths and neuron activities
        - stack: rewards and work costs are tracked for every step
        """    

        _vectorized_env_state, _synapse_strengths, _policy_params, _neuron_activities = carry

        _sensory_input_nn = jnp.concatenate(
            [_vectorized_env_state.observations[label] for label in config["environment"]["sensor_selection"]],
            axis = 1
        )     
        
        if config["controller"]["hebbian"] == True:
            # apply a Hebbian control: updates synapse strengths using learning rules, yields action and neuron activities
            _action, _synapse_strengths, _neuron_activities = vectorized_controller_apply(
                                                                    _sensory_input_nn,
                                                                    _synapse_strengths,
                                                                    _policy_params,
                                                                    _neuron_activities
                                                                    )
        else:
            # apply a static control: just yields action and neuron activities
            _action, _neuron_activities = vectorized_controller_apply(
                                                                    _sensory_input_nn,
                                                                    _synapse_strengths
                                                                    )
        
        _vectorized_env_state_updated = vectorized_env_step(state=_vectorized_env_state, action=_action)
        
        reward_step = _vectorized_env_state_updated.reward
        cost_step = cost_step_during_rollout(env_state_observations=_vectorized_env_state_updated.observations, cost_expr=config["evolution"]["cost_expr"])
        penal_step = penal_step_during_rollout(env_state_observations=_vectorized_env_state_updated.observations, penal_expr=config["evolution"]["penal_expr"])

        carry = [_vectorized_env_state_updated, _synapse_strengths, _policy_params, _neuron_activities]
        return carry, [reward_step, cost_step, penal_step]

    final_carry, stack = jax.lax.scan(
        step,
        carry_init,
        None,
        controller.env_container.environment_configuration.total_num_control_steps,
    )

    rewards, costs, penal = stack
    vectorized_env_state_final = final_carry[0]
    return vectorized_env_state_final, (rewards, costs, penal), rng
# jit the rollout function
rollout = jax.jit(rollout, static_argnames=("env", "controller", "parallel_dim"))



def rollout_old(
        mjx_vectorized_env: MJXEnv,
        nn_model: ExplicitMLP,
        policy_params_shaped: dict,
        total_num_control_timesteps: int,
        sensor_selection: tuple,
        rng: chex.PRNGKey,
        NUM_MJX_ENVIRONMENTS: int,
        cost_expr: str = "nocost",
        penal_expr: str = "nopenal",
        target_position = None
        ):
    """
    Do a single episode rollout
    Inputs:
    - vectorized format of parallel mjx environments that can be run
    - Reshaped policy params to match neural network synapses and biases
    - Total_num_control_timesteps
    - sensor selection
    - rng
    - Total number of parallel MJX environments
    - cost_expr: string describing the way costs are calculated
    - target_position: list of 3 numbers: XYZ coordinates of the target
    Outputs:
    - Final MJX environment state (vectorized)
    - Rewards: sum of all rewards of every timestep
    - rng
    """

    vectorized_nn_model_apply = jax.jit(jax.vmap(nn_model.apply))
    mjx_vectorized_step = jax.jit(jax.vmap(mjx_vectorized_env.step))
    mjx_vectorized_reset = jax.jit(jax.vmap(mjx_vectorized_env.reset))
    rng, mjx_vectorized_env_rng = jax.random.split(rng, 2)
    mjx_vectorized_env_rng = jnp.array(jax.random.split(mjx_vectorized_env_rng, NUM_MJX_ENVIRONMENTS))
    if target_position:
        mjx_vectorized_state_reset = mjx_vectorized_reset(rng=mjx_vectorized_env_rng, target_position=target_position)
    else:
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
        
        
        _action = vectorized_nn_model_apply(_policy_params_shaped, _sensory_input_nn)
        
        _mjx_vectorized_state_updated = mjx_vectorized_step(state=_mjx_vectorized_state, action=_action)
        
        reward_step = _mjx_vectorized_state_updated.reward


        cost_step = cost_step_during_rollout(env_state_observations=_mjx_vectorized_state.observations, cost_expr=cost_expr)
        penal_step = penal_step_during_rollout(env_state_observations=_mjx_vectorized_state.observations, penal_expr=penal_expr)


        # joint_velocities = _mjx_vectorized_state.observations['joint_velocity']
        # torques = _mjx_vectorized_state.observations['joint_actuator_force']

        # assert cost_expr in ["nocost", "torque x angvel", "torque"], 'cost_expr must be chosen from ["nocost", "torque x angvel", "torque"]'
        # if cost_expr == "nocost":
        #     cost_step = 0
        # elif cost_expr == "torque x angvel":
        #     cost_step = jnp.sum(jnp.abs(joint_velocities)*jnp.abs(torques), axis = 1)
        # elif cost_expr == "torque":
        #     cost_step = jnp.sum(jnp.abs(torques), axis = 1)
        
        # above sum sums over all the segments of 1 single individual in the generation
        # summing this power over every time step results in the total work done (done outside of the rollout function)

        carry = [_mjx_vectorized_state_updated, _policy_params_shaped]
        return carry, [reward_step, cost_step, penal_step]

    final_carry, stack = jax.lax.scan(
        step,
        carry_init,
        None,
        total_num_control_timesteps
    )

    rewards, costs, penal = stack
    
    mjx_vectorized_state_final = final_carry[0]
    return mjx_vectorized_state_final, (rewards, costs, penal), rng
# jit the rollout function
rollout_old = jax.jit(rollout_old, static_argnames=("mjx_vectorized_env", "nn_model",  "total_num_control_timesteps", "NUM_MJX_ENVIRONMENTS", "sensor_selection", "cost_expr", "penal_expr", "target_position"))



def cost_step_during_rollout(
        env_state_observations: dict,
        cost_expr: str
):
    assert cost_expr in ["nocost", "torque x angvel", "torque"], 'cost_expr must be chosen from ["nocost", "torque x angvel", "torque"]'
    torques = env_state_observations['joint_actuator_force']
    joint_velocities = env_state_observations['joint_velocity']
    # torques or joint_velocities dim can be (popsize, 2*num_segments) or (popsize, 2*num_segments, timesteps) depending on whether it is an observation stack or not
        
    if cost_expr == "nocost":
        cost_step = jnp.zeros_like(jnp.sum(jnp.abs(torques), axis = 1)) # we know jnp.sum(jnp.abs(torques), axis = 1) has good dims so make zeros array like it
    elif cost_expr == "torque x angvel":
        cost_step = jnp.sum(jnp.abs(joint_velocities)*jnp.abs(torques), axis = 1)
    elif cost_expr == "torque":
        cost_step = jnp.sum(jnp.abs(torques), axis = 1)
    else:
        raise "No cost steps have been generated: check whether correct cost_expr has been supplied"

    return cost_step

def penal_step_during_rollout(
        env_state_observations: dict,
        penal_expr: str

):
    assert penal_expr in ["nopenal", "body_stability"], 'penal_expr must be chosen from ["nopenal", "body_stability", "torque"]'
    if penal_expr == "nopenal":
        penal_step = jnp.zeros_like(jnp.sum(jnp.abs(env_state_observations['joint_actuator_force']), axis = 1)) # we know jnp.sum(jnp.abs(torques), axis = 1) has good dims so make zeros array like it
    elif penal_expr == "body_stability":
        # env_state_observations["disk_rotation"] dim can be (popsize, 3) or (popsize, 3, timesteps) depending on whether it is an observation stack or not
        rx = jnp.take(env_state_observations["disk_rotation"], 0, axis = 1)  # slicing env_state_observations["disk_rotation"][:,0] but without nowing how many dims there are in total
        ry = jnp.take(env_state_observations["disk_rotation"], 1, axis = 1) 
        tz = jnp.take(env_state_observations["disk_rotation"], 2, axis = 1)
        norm = jnp.linalg.norm(jnp.array([rx, ry, tz]), axis = 0)
        penal_step = jnp.exp(norm) # taking exponential makes sure that when penal comes in denominator of the fitness,
        # the fitness still converges also for penal = 0
    else:
        raise "No penal steps have been generated: check whether correct penal_expr has been supplied"

    return penal_step


def generate_video_joint_angle_raw(
        policy_params_to_render,
        param_reshaper: ParameterReshaper,
        rng: chex.PRNGKey,
        mjx_vectorized_env: MJXEnv,
        sensor_selection: Sequence[str],
        arm_setup: Sequence[int],
        nn_model: ExplicitMLP,
        damage: bool = False,
        arm_setup_damage: Sequence[int] = None,
        visualise_increasing_opacity: bool = False
):
    """
    if visualise_increasing_opacity = True: 2 additional arrays returned:
    - background_frame: numpy array of an image of the fully opaque background
    - brittle_star_frames: list of numpy arrays of only the brittle stars
    """
    if damage == True:
        assert arm_setup_damage, "provide an arm_setup_damage"
        check_damage(arm_setup, arm_setup_damage)

    policy_params_solution_shaped = param_reshaper.reshape(policy_params_to_render)
    NUM_MJX_ENVIRONMENTS_render = policy_params_to_render.shape[0]

    rng, mjx_vectorized_env_rng = jax.random.split(rng, 2)
    mjx_vectorized_env_rng = jnp.array(jax.random.split(mjx_vectorized_env_rng, NUM_MJX_ENVIRONMENTS_render))

    mjx_vectorized_step = jax.jit(jax.vmap(mjx_vectorized_env.step))
    mjx_vectorized_reset = jax.jit(jax.vmap(mjx_vectorized_env.reset))

    vectorized_nn_model_apply = jax.jit(jax.vmap(nn_model.apply))

    mjx_vectorized_state = mjx_vectorized_reset(rng=mjx_vectorized_env_rng)

    frames = []
    joint_angles_ip = []
    joint_angles_oop = []
    if visualise_increasing_opacity:
        brittle_star_frames = []
        env_state_background = move_camera(state=mjx_vectorized_state)
        env_state_background = change_alpha(state = env_state_background, brittle_star_alpha=0.0, background_alpha=1.0)
        background_frame = post_render(
            mjx_vectorized_env.render(state=env_state_background),
            mjx_vectorized_env.environment_configuration
            )
        
    while not jnp.any(mjx_vectorized_state.terminated | mjx_vectorized_state.truncated):
        
        sensory_input = jnp.concatenate(
            [mjx_vectorized_state.observations[label] for label in sensor_selection],
            axis = 1
        )

        if damage:
            sensory_input = pad_sensory_input(sensory_input, arm_setup, arm_setup_damage, sensor_selection)

        joint_angles_ip_t = []
        joint_angles_oop_t = []
        j = 0

        if damage:
            arms = arm_setup_damage
        else:
            arms = arm_setup

        for n in arms:
            if n != 0:
                joint_angles_ip_t.append(mjx_vectorized_state.observations["joint_position"][0][j*2*n:(j+1)*2*n:2])
                joint_angles_oop_t.append(mjx_vectorized_state.observations["joint_position"][0][j*2*n+1:(j+1)*2*n+1:2])
                j += 1

        joint_angles_ip.append(joint_angles_ip_t)
        joint_angles_oop.append(joint_angles_oop_t)

        action = vectorized_nn_model_apply(policy_params_solution_shaped, sensory_input)
        if damage:
            action = select_actuator_output(action, arm_setup, arm_setup_damage)
        
        mjx_vectorized_state = mjx_vectorized_step(state=mjx_vectorized_state, action=action)
        
        frames.append(
            post_render(
                mjx_vectorized_env.render(state=mjx_vectorized_state),
                mjx_vectorized_env.environment_configuration
                )
            )
        
        if visualise_increasing_opacity:
            env_state_brittle_star = move_camera(state=mjx_vectorized_state)
            env_state_brittle_star = change_alpha(state = env_state_brittle_star, brittle_star_alpha=1.0, background_alpha=0.0)
            brittle_star_frames.append(
                post_render(
                mjx_vectorized_env.render(state=env_state_brittle_star),
                mjx_vectorized_env.environment_configuration
                )
            )
    
    if visualise_increasing_opacity:
        return frames, joint_angles_ip, joint_angles_oop, background_frame, brittle_star_frames
    else:
        return frames, joint_angles_ip, joint_angles_oop


