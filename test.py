import numpy as np
import matplotlib.pyplot as plt

a = 1
b = 2
c = 0.2

d = -1
e = -0.2
f = 2

x = np.array([a,b,c])
y = np.array([d,e,f])
z = np.array([x,y])

print(np.all(x == y))
print(np.all(x == z[0]))

print(z)

print(np.linalg.norm(z, axis = 0))
print(np.linalg.norm(z, axis = 1))






