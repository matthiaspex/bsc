import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pickle




with open("example_kernels.pkl", "rb") as fp:
    kernels = pickle.load(fp)

np.set_printoptions(precision=2)

for i in range(len(kernels)):
    if i%10 == 0:
        print("min: ", np.min(kernels[i]["params"][f"layers_{0}"]["kernel"][0]))
        print("max: ", np.max(kernels[i]["params"][f"layers_{0}"]["kernel"][0]))




num_layers = len(kernels[0]["params"])
width_ratios = np.zeros(num_layers)
for i in range(num_layers):
    width_ratios[i] = kernels[0]["params"][f"layers_{i}"]["kernel"][0].shape[-1]
print(width_ratios)
width_ratios = width_ratios/np.sum(width_ratios)*len(width_ratios)
print(width_ratios)


fig, axes = plt.subplots(1, num_layers, figsize = (5*num_layers, 5), width_ratios=width_ratios)
fig.tight_layout(pad=3)                  

ims = []
cbs = []
for i in range(num_layers):
    ims.append(f"im_{i}")
    cbs.append(f"cb_{i}")
    globals()[ims[i]] = axes[i].imshow(kernels[0]["params"][f"layers_{i}"]["kernel"][0], cmap="seismic",\
                                       interpolation="nearest")
    globals()[cbs[i]] = fig.colorbar(globals()[ims[i]], orientation="vertical")
    axes[i].set_title(f"layer {i}")
    axes[i].set_xlabel('post-synaptic nodes')
    axes[i].set_ylabel('pre-synaptic nodes')

def update(num, kernels):
    num_layers = len(kernels[num]["params"])
    for i in range(num_layers):
        arr = kernels[num]["params"][f"layers_{i}"]["kernel"][0]
        vmax = np.max(arr)
        vmin = np.min(arr)
        globals()[ims[i]].set_data(arr)
        globals()[ims[i]].set_clim(vmin, vmax)

fps = 25

ani = animation.FuncAnimation(fig=fig, func=update, frames=len(kernels), fargs=[kernels])
ani.save(filename="example_kernels.mp4", writer="ffmpeg", fps=fps) # either "ffmpeg" or "imagemagick if you want gifs"









