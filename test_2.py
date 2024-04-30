import pickle
import jax
from jax import numpy as jnp

from bsc_utils.simulate.base import cost_step_during_rollout

with open('test_obs.pkl', 'rb') as fp:
    obs = pickle.load(fp)


# with open('test_rewards.pkl', 'rb') as fp:
#     rewards = pickle.load(fp)

# print(rewards.shape)
# print(rewards)

# rewards_exp = jnp.expand_dims(rewards, axis = -1)

# for i in range(5):
#     rewards_exp = jnp.concatenate(
#         [rewards_exp, jnp.expand_dims(rewards, axis = -1)],
#         axis = -1)

# print(rewards_exp.shape)
# print(rewards_exp)

print(obs.keys())


print(jax.tree_util.tree_map(lambda x: x.shape, obs))

obs_merge = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis = -1), obs)



for i in range(3):
    obs_merge = jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate(
            [x, jnp.expand_dims(y, axis = -1)],
            axis=-1),
        obs_merge, obs)


print(obs_merge["disk_angular_velocity"])

print(jax.tree_util.tree_map(lambda x: x.shape, obs_merge))


cost_step = cost_step_during_rollout(obs_merge, "torque x angvel")

print(cost_step)
print(cost_step.shape)












