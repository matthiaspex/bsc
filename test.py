
import jax
from jax import numpy as jnp
import sys
from bsc_utils.analyze.episode import Simulator
from bsc_utils.miscellaneous import load_config_from_yaml
import pickle

rng = jax.random.PRNGKey(0)

config = load_config_from_yaml("trained_policy_params/b01/b01_r03 distance arms [5, 5, 5, 5, 5] popsize 6912 torque reward _ cost.yaml")

simulator = Simulator(config)
simulator.generate_env()
simulator.generate_episode_data_undamaged(jax.random.PRNGKey(0))

sys.exit()
rng, vectorized_env_rng = jax.random.split(rng, 2)
vectorized_env_rng = jnp.array(jax.random.split(vectorized_env_rng, 2))
vectorized_env_reset = jax.jit(jax.vmap(simulator.env.reset))
vectorized_env_state = vectorized_env_reset(rng=vectorized_env_rng)

# observations = vectorized_env_state.observations
# reward = vectorized_env_state.reward

# with open('test_rewards.pkl', 'wb') as fp:
#     pickle.dump(reward, fp)





