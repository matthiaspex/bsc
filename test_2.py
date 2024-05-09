from bsc_utils.miscellaneous import get_target_positions
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

rng = jax.random.PRNGKey(2)
parallel_dim = 2


target_positions = get_target_positions(rng,
                                             distance=5.0, 
                                             num_rowing=5,
                                             num_reverse_rowing=5,
                                             num_random_positions=0,
                                             parallel_dim=2,
                                             parallel_constant=True)


print(len(target_positions))
print(target_positions[0])
print(target_positions[0].shape)


x_targets =[]
y_targets = []
for target in target_positions:
    x_targets.append(target[0,0])
    y_targets.append(target[0,1])

x = jnp.linspace(-5.,5.,1001)
y1 = jnp.sqrt(25-x**2)
y2 = -jnp.sqrt(25-x**2)

plt.scatter(x_targets, y_targets)
plt.plot(x,y1, 'r')
plt.plot(x,y2, 'r')
plt.show()