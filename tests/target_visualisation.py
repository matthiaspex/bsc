from bsc_utils.miscellaneous import get_target_positions
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

rng = jax.random.PRNGKey(2)
parallel_dim = 1
num_rowing = 2
num_reverse_rowing = 2
num_random_positions = 0

target_positions = get_target_positions(rng,
                                             distance=3.0, 
                                             num_rowing=num_rowing,
                                             num_reverse_rowing=num_reverse_rowing,
                                             num_random_positions=num_random_positions,
                                             parallel_dim=parallel_dim,
                                             parallel_constant=True)


assert target_positions[0].shape == (parallel_dim, 3), "targets input is wrong dimension. Make sure it has dim (parallel_dim, 3)"

print(len(target_positions))
print(target_positions[0])
print(target_positions[0].shape)


x_targets =[]
y_targets = []
for target in target_positions:
    x_targets.append(target[0,0]*29+113)
    y_targets.append(target[0,1]*29+113)


x = jnp.linspace(-3.,3.,1001)
y1 = jnp.sqrt(9-x**2)
y2 = -jnp.sqrt(9-x**2)

a = jnp.linspace(113.5-85,113.5+85,1001)
b1 = jnp.sqrt(85**2-(x-113.5)**2)+113.5
b2 = -jnp.sqrt(85**2-(x-113.5)**2)+113.5

im = plt.imread("Images//tmp//arena 4x4 with morphology.png")
plt.imshow(im)
plt.rcParams["figure.figsize"] = (5,5)
circle1= plt.Circle((113.5,113.5), 87, color='dimgrey', fill=False)
plt.gca().add_patch(circle1)
# plt.scatter([0],[0],[10], color='k', marker='*', )
plt.scatter(x_targets[:num_rowing], y_targets[:num_rowing], color = 'green')
plt.scatter(x_targets[num_rowing:num_rowing+num_reverse_rowing], y_targets[num_rowing:num_rowing+num_reverse_rowing], color = 'red')

# plt.plot(x,y1, color='dimgrey')
# plt.plot(x,y2, color='dimgrey')

# plt.plot(a,b1, color='dimgrey')
# plt.plot(a,b2, color='dimgrey')
plt.show()