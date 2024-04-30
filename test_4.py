import jax
from jax import numpy as jnp

def powers(x):
    seq = []
    for i in range(4):
        seq.append(x**i)
    return seq

powers_vect = jax.jit(jax.vmap(powers))

x = jnp.array([1,2,3])

x_2d = jnp.array([[1,2,3],[4,5,6]])

x_vect = jnp.stack([x_2d, x_2d, x_2d])

print(x)
print(x_vect)
print(x_vect.shape)

print(powers(x))
print(powers(x_vect)[0])
print(powers_vect(x_vect)[0])


