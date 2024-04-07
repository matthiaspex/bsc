from jax import numpy as jnp

print(jnp.zeros(5))


a = jnp.array([1,2,3,4,5])
b = jnp.array([6,7,8,9,10])

c = jnp.stack([a,b])
print(c)

d = jnp.stack([c,c,c])
print(d.shape)
print(d)


e = d[:,1,:]
print(e)

f = jnp.take(d, 1, axis = 1)

print(f)

print(e-f)