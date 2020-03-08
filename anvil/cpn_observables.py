import torch
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import use
from math import pi

use("TkAgg")


def free_energy(sample_training_output, action, beta):
    S = action(sample_training_output)
    print(f"Action mean: {torch.mean(S)}, std: {torch.std(S)}")
    Z = torch.mean(torch.exp(-S))
    print(f"Partition function: {Z}")
    coeff = -1.0 / beta
    print(f"Free energy: {coeff * torch.log(Z)}")
    return coeff * torch.log(Z)


def plot_dists(sample_training_output, action, training_geometry, n_coords):

    # Action
    fig, ax = plt.subplots()
    S = action(sample_training_output)
    action_hist, action_bins = np.histogram(S, bins=50)
    action_w = action_bins[1] - action_bins[0]
    action_c = (action_bins[:-1] + action_bins[1:]) / 2
    ax.bar(action_c, action_hist, width=action_w, color="w", edgecolor="r")
    ax.set_title("Action")
    fig.tight_layout()
    fig.savefig("action_distribution.png")

    # angles
    sample = sample_training_output.view(-1, training_geometry.length ** 2, n_coords)

    if n_coords > 3:
        fig2, all_ax = plt.subplots((n_coords + 1) // 2, 2)
        all_ax_1d = [ax for tup in all_ax for ax in tup]
    else:
        fig2, all_ax = plt.subplots(1, n_coords)
        all_ax_1d = all_ax
    for i, ax in enumerate(all_ax_1d):
        phi = sample[:, 0, i]  # just take one lattice site (translation invariance)
        phi_hist, phi_bins = np.histogram(phi, bins=50)
        phi_w = phi_bins[1] - phi_bins[0]
        phi_c = (phi_bins[:-1] + phi_bins[1:]) / 2
        ax.bar(phi_c, phi_hist, width=phi_w, color="w", edgecolor="g")
        
        #if i == n:
        #    ax.set_xlim(left=-pi, right=pi)
        #else:
        #    ax.set_xlim(left=-pi/2, right=pi/2)

        ax.set_title(rf"$\phi_{i+1}(0)$")

    fig2.tight_layout()
    fig2.savefig("phi_distributions.png")
    
    #plt.show()



