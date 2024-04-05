import chex
import jax
from jax import numpy as jnp
from moojoco.environment.mjx_env import MJXEnv, MJXEnvState
from bsc_utils.controller import ExplicitMLP
from typing import Sequence
from evosax import ParameterReshaper
from bsc_utils.visualization import post_render
from bsc_utils.damage import pad_sensory_input, select_actuator_output, check_damage

def rollout(
        mjx_vectorized_env: MJXEnv,
        nn_model: ExplicitMLP,
        policy_params_shaped: dict,
        total_num_control_timesteps: int,
        sensor_selection: tuple,
        rng: chex.PRNGKey,
        NUM_MJX_ENVIRONMENTS: int,
        cost_expr: str = "nocost",
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

        joint_velocities = _mjx_vectorized_state.observations['joint_velocity']
        torques = _mjx_vectorized_state.observations['joint_actuator_force']

        assert cost_expr in ["nocost", "torque x angvel", "torque"], 'cost_expr must be chosen from ["nocost", "torque x angvel", "torque"]'
        if cost_expr == "nocost":
            cost_step = 0
        elif cost_expr == "torque x angvel":
            cost_step = jnp.sum(jnp.abs(joint_velocities)*jnp.abs(torques), axis = 1)
        elif cost_expr == "torque":
            cost_step = jnp.sum(jnp.abs(torques), axis = 1)
        
        # above sum sums over all the segments of 1 single individual in the generation
        # summing this power over every time step results in the total work done (done outside of the rollout function)

        carry = [_mjx_vectorized_state_updated, _policy_params_shaped]
        return carry, [reward_step, cost_step]

    final_carry, stack = jax.lax.scan(
        step,
        carry_init,
        None,
        total_num_control_timesteps
    )

    rewards, costs = stack
    
    mjx_vectorized_state_final = final_carry[0]
    return mjx_vectorized_state_final, rewards, costs, rng




def generate_video_joint_angle_raw(
        policy_params_to_render,
        param_reshaper: ParameterReshaper,
        rng: chex.PRNGKey,
        mjx_vectorized_env: MJXEnv,
        sensor_selection: Sequence[str],
        arm_setup: Sequence[int],
        nn_model: ExplicitMLP,
        damage: bool = False,
        arm_setup_damage: Sequence[int] = None
):
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
    
    return frames, joint_angles_ip, joint_angles_oop