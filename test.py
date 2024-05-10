from bsc_utils.miscellaneous import get_target_positions
import jax


a = [5,4,1,9,626,6,3,0.2]
print(min(a))
print(max(a))

rng = jax.random.PRNGKey(0)
rng, rng_targets_simulator = jax.random.split(rng, 2)
targets_simulator = get_target_positions(rng=rng_targets_simulator,
                                distance=7,
                                num_rowing=0,
                                num_reverse_rowing=0,
                                num_random_positions=1,
                                parallel_dim=1,
                                parallel_constant=True)
print(targets_simulator)
print(targets_simulator[0].shape)