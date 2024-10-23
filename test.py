import matplotlib.pyplot as plt
import numpy as np

# Create some data to plot
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

# Set up the figure and axes
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

# Plot on each subplot
axes[0].plot(x, y1, color='b', label='sin(x)')
axes[0].set_title('Sine Function')
axes[0].set_xlabel('x')
axes[0].set_ylabel('sin(x)')
axes[0].grid(True)

axes[1].plot(x, y2, color='r', label='cos(x)')
axes[1].set_title('Cosine Function')
axes[1].set_xlabel('x')
axes[1].set_ylabel('cos(x)')
axes[1].grid(True)

axes[2].plot(x, y3, color='g', label='tan(x)')
axes[2].set_title('Tangent Function')
axes[2].set_xlabel('x')
axes[2].set_ylabel('tan(x)')
axes[2].grid(True)

# Adjust layout to prevent overlap and make it look nice
plt.tight_layout()

# Display the plot
plt.show()
