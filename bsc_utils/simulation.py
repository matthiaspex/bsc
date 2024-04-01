import chex
import jax
from jax import numpy as jnp
from moojoco.environment.mjx_env import MJXEnv, MJXEnvState
from bsc_utils.controller import ExplicitMLP

def rollout(
        mjx_vectorized_env: MJXEnv,
        nn_model: ExplicitMLP,
        policy_params_shaped: dict,
        total_num_control_timesteps: int,
        sensor_selection: tuple,
        rng: chex.PRNGKey,
        NUM_MJX_ENVIRONMENTS: int,
        cost_expr: str = "no_cost"
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
    - string describing the way costs are calculated
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
        if cost_expr == "no_cost":
            cost_step = 0
        elif cost_expr == "torque*ang_vel":
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