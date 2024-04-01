from typing import Sequence

def reward_cost_to_fitness(
        total_reward: Sequence[float],
        total_cost: Sequence[float],
        fitness_expr: str
        ):
    assert fitness_expr in ["reward", "reward/cost", "reward+reward/cost"], 'fitness_expr must be chosen from ["reward", "reward/cost", "reward+reward/cost"]'
    if fitness_expr == "reward":
        fitness = total_reward
    if fitness_expr == "reward/cost":
        fitness = total_reward/total_cost
    elif fitness_expr == "reward+reward/cost":
        fitness = total_reward + total_reward/total_cost
    return fitness