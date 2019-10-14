#!/usr/bin/env python
"""
train_example.py

Toy training example where network is trained on a small multigaussian system
and we check if the covariance of the target is reproduced.

Should provide context to help integrate the neural networks into more
interesting case of 2D lattice configurations

"""
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from norm_flow_pytorch import NormalisingFlow, shifted_kl

L = 2 # very small system
N_UNITS = L**2

np.random.seed(seed=0)
A_NP = np.array(np.random.rand(N_UNITS, N_UNITS), dtype=np.float32)
A = torch.from_numpy(A_NP)
# just make some positive definite matrix
COV_TARGET = A@A.T

INV = torch.inverse(COV_TARGET)

def target_distribution_s(phi):
    r"""Returns action S(\phi) for a stack of states \phi, shape (N_states, D).
    The action corresponds to a toy training example of a multigaussian
    distribution.
    """
    return 0.5*((phi@INV)*phi).sum(axis=-1, keepdim=True)

def split_transformation(length, a_left=True):
    r"""Given a flattened 2D state, represented by a vector \phi, returns a
    matrix transformation, M, which acting on the flattened state seperates
    two halves of the matrix according to a checkerboard pattern into
    \phi = (\phi_a, \phi_b) (by default even sites sent left, odd sites sent
    right). This behaviour can be changed with `a_left` flag.

    """
    if a_left:
        condition = (1, 0)
    else:
        condition = (0, 1)
    N = length**2
    state = np.zeros((length, length)) # define a checkerboard
    state[1::2, 1::2] = 1
    state[::2, ::2] = 1 # even sites are = 1
    flat = state.flatten()
    left = np.zeros((N, N), dtype=np.float32)
    right = np.zeros((N, N), dtype=np.float32)
    # ceil lets this handle length odd, unneccesary for our project
    left[np.arange(np.ceil(N/2), dtype=int), np.where(flat == condition[0])[0]] = 1.
    right[np.arange(np.ceil(N/2), N, dtype=int), np.where(flat == condition[1])[0]] = 1.
    return torch.from_numpy(left), torch.from_numpy(right)

N_BATCH = 2000

def train_multivariate(model, epochs, a_left, b_right):
    """example of training loop of model"""
    # create your optimizer and a scheduler
    optimizer = optim.Adadelta(model.parameters(), lr=1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500)

    # let's use tqdm to see progress
    pbar = tqdm(range(epochs), desc=f"loss: N/A")
    for i in pbar:
        # gen simple states
        z = torch.randn((N_BATCH, N_UNITS))
        phi = model.inverse_map(z)
        # phi here will be phi = (phi_a, phi_b) so need to undo transform
        # phi@a_left = (a_left.T @ phi.T).T
        target = target_distribution_s(phi@a_left + phi@b_right)
        output = model(phi)

        model.zero_grad() # get rid of stored gradients
        loss = shifted_kl(output, target)
        loss.backward() # calc gradients

        optimizer.step()
        scheduler.step(loss)

        if (i%50) == 0:
            pbar.set_description(f"loss: {loss.item()}")

def sample_multivariate(model, a_left, b_right, n_large):
    """Analyse and plot if the covariance has been reproduced"""
    with torch.no_grad(): # don't want gradients being tracked in sampling stage
        z = torch.randn((n_large, N_UNITS))
        phi = model.inverse_map(z)
        a = phi@a_left + phi@b_right
        cov = np.cov(a, rowvar=False)
    # Save plots
    fig, ax = plt.subplots()
    im = ax.imshow(cov)
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    ax.set_title("Covariance matrix sampled from model.")
    fig.tight_layout()
    fig.savefig("example_output/test.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    im = ax.imshow(COV_TARGET)
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    ax.set_title("Target covariance matrix.")
    fig.tight_layout()
    fig.savefig("example_output/targ.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    im = ax.imshow(cov/np.array(COV_TARGET))
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    ax.set_title(
        "Ratio of sampled model to target covariance matrix."
    )
    fig.tight_layout()
    fig.savefig("example_output/ratio.png")
    plt.close(fig)

def main():
    """train and sample model on toy distirbution"""
    # set seed, hopefully result is reproducible
    torch.manual_seed(0)
    # define simple mode, each network is single layered
    model = NormalisingFlow(
        size_in=N_UNITS, n_affine=8, affine_hidden_shape=(16,)
    )
    epochs = 5000 # Gives a decent enough approx.
    a_left, b_right = split_transformation(L)
    # model needs to learn rotation and rescale
    train_multivariate(model, epochs, a_left, b_right)
    n_large = 20000 # to estimate covmat
    sample_multivariate(model, a_left, b_right, n_large)

if __name__ == "__main__":
    main()
