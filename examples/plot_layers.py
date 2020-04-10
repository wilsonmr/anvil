import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt

n_coords = 2    # angles per lattice site
L = 4           # lattice length
n_fig = 4       # number of figures to plot (should be a square number)

V = L * L
l_fig = int(sqrt(n_fig))

base = np.loadtxt("base.txt").reshape(-1, V, n_coords)
target = np.loadtxt("target.txt").reshape(-1, V, n_coords)
#projected = np.loadtxt("projected.txt").reshape(-1, V, n_coords)

layers = []
n_lay = 0
while True:
    try:
        new_layer = np.loadtxt(f"layer_{n_lay}.txt")
    except:
        break
    layers.append(new_layer.reshape(-1, V, n_coords))
    n_lay += 1

tanbins = np.tan(np.linspace(-0.5 * pi + 0.1, 0.5 * pi - 0.1, 51))

checker = np.zeros((L, L), dtype=bool)
checker[::2, ::2] = True
checker[1::2, 1::2] = True

ind = np.arange(V).reshape(L, L)
flat_split_ind = np.zeros(V, dtype=int)
flat_split_ind[:V//2] = ind[checker]
flat_split_ind[V//2:] = ind[~checker]

flat_ind = [np.where(flat_split_ind == i)[0][0] for i in range(V)]

# Instead of plotting every lattice site, just plot the top corner
corner_ind = []
for row in range(l_fig):
    corner_ind += [(L*row + i) for i in range(l_fig)]
flat_ind_to_plot = [flat_ind[i] for i in corner_ind]

for i in range(n_coords):
    fig, axes = plt.subplots(l_fig, l_fig, figsize=(10,10))

    for j, ax in zip(flat_ind_to_plot, axes.flatten()):

        ax.hist(base[:, j, i], bins=50, density=True, histtype="step", label="base")
        #ax.hist(projected[:, j, i], bins=tanbins, density=True, histtype="step", label="projected")
        
        if j < V//2: k = 0
        else: k = 1
        for l in range(k, n_lay, 2):
            ax.hist(layers[l][:, j, i], bins=tanbins, density=True, histtype="step", label=f"layer {l}")
        
        ax.hist(target[:, j, i], bins=50, density=True, histtype="step", label="output")
        ax.legend()
    
    plt.tight_layout()
    plt.show()


