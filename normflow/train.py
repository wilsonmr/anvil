"""
train.py

Module containing functions required to train model
"""

from tqdm import tqdm

import torch
import torch.optim as optim

N_BATCH = 2000  # keep batch size constant for now


def shifted_kl(log_tilde_p: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    r"""Sample mean of the shifted Kullbach-Leibler divergence between target
    and model distribution.

    Parameters
    ----------
    log_tilde_p: torch.Tensor
        column of log (\tilde p) for a sample of states, which is returned by
        forward pass of `NormalisingFlow` model
    action: torch.Tensor
        column of actions S(\phi) for set of sample states

    Returns
    -------
    out: torch.Tensor
        torch tensor with single element, corresponding to the estimation of the
        shifted K-L for set of sample states.

    """
    return torch.mean(log_tilde_p + action, dim=0)


def train(model, action, *, epochs, save_int, n_batch, outpath):
    """example of training loop of model"""
    # create your optimizer and a scheduler
    optimizer = optim.Adadelta(model.parameters(), lr=1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500)
    # let's use tqdm to see progress
    pbar = tqdm(range(epochs), desc=f"loss: N/A")
    n_units = model.size_in
    for i in pbar:
        # gen simple states
        z = torch.randn((N_BATCH, n_units))
        phi = model.inverse_map(z)
        target = action(phi)
        output = model(phi)

        model.zero_grad()  # get rid of stored gradients
        loss = shifted_kl(output, target)
        loss.backward()  # calc gradients

        optimizer.step()
        scheduler.step(loss)

        if (i % 50) == 0:
            pbar.set_description(f"loss: {loss.item()}")
