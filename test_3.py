import jax
from jax import numpy as jnp



print(jnp.zeros(5))


a = jnp.array([1,2,3,4,5])
b = jnp.array([6,7,8,9,10])

c = jnp.stack([a,b])
print(c)

d = jnp.stack([c,c,c])
print(d.shape)
print(d)
print(jnp.size(d))

e = (d, a, b)

sizes = []
print(e)
jax.tree_util.tree_map(lambda x: sizes.append(jnp.size(x[0])), e)
print(sum(sizes))


try:
    print(a)
except NameError:
    print("variable is not defined")
