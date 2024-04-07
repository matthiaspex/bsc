from typing import Sequence
from jax import numpy as jnp

def efficiency_from_reward_cost(
        total_reward: Sequence[float],
        total_cost: Sequence[float],
        efficiency_expr: str
        ):
    assert efficiency_expr in ["reward", "reward _ cost", "reward + reward _ cost"], 'efficiency_expr must be chosen from ["reward", "reward _ cost", "reward + reward _ cost"]'
    if efficiency_expr == "reward" or jnp.all(total_cost == 0):
        efficiency = total_reward
    if efficiency_expr == "reward _ cost":
        efficiency = total_reward/total_cost
    elif efficiency_expr == "reward + reward _ cost":
        efficiency = total_reward + total_reward/total_cost
    return efficiency


def fitness_from_stacked_data(
        stacked_data,
        efficiency_expr: str
):
    """"
    Stacked data is a tuple consisting of (rewards, costs, penal)
    Each of those is an array with dims [num_control_steps, num_parallel_envs] (e.g. (125, 6912))
    Summing those over num_control_steps results in complete reward, cost and penal for the entire simulation
    """
    rewards, costs, penal = stacked_data

    total_reward = jnp.sum(rewards, axis = 0)
    total_cost = jnp.sum(costs, axis = 0)
    total_penal = jnp.sum(penal, axis = 0)
    efficiency = efficiency_from_reward_cost(total_reward, total_cost, efficiency_expr)

    if jnp.all(total_penal == 0):
        fitness = efficiency
    else:
        fitness = efficiency/total_penal
    
    return fitness


