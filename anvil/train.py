"""
train.py

Module containing functions required to train model
"""

from tqdm import tqdm

import torch
import torch.optim as optim

import numpy as np

def shifted_kl(
    model_log_density: torch.Tensor, target_log_density: torch.Tensor
) -> torch.Tensor:
    r"""Sample mean of the shifted Kullbach-Leibler divergence between target
    and model distribution.

    Parameters
    ----------
    model_log_density: torch.Tensor
        column of log (\tilde p) for a sample of states, which is returned by
        forward pass of `RealNVP` model
    target_log_density: torch.Tensor
        column of log(p) for a sample of states, which includes the negative
        action -S(\phi) and a possible contribution to the volume element due
        to a non-trivial parameterisation.

    Returns
    -------
    out: torch.Tensor
        torch tensor with single element, corresponding to the estimation of the
        shifted K-L for set of sample states.

    """
    return torch.mean(model_log_density - target_log_density, dim=0)


def train(
    loaded_model,
    base,
    target,
    *,
    train_range,
    save_interval,
    n_batch=2000,
    outpath,
    current_loss,
    loaded_optimizer,
    scheduler_kwargs={"patience": 500},
):
    """training loop of model"""
    # create your optimizer and a scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        loaded_optimizer, **scheduler_kwargs
    )
    # let's use tqdm to see progress
    pbar = tqdm(range(*train_range), desc=f"loss: {current_loss}")
    n_units = base.size_out
    for i in pbar:
        if (i % save_interval) == 0:
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": loaded_model.state_dict(),
                    "optimizer_state_dict": loaded_optimizer.state_dict(),
                    "loss": current_loss,
                },
                f"{outpath}/checkpoint_{i}.pt",
            )
        # gen simple states
        z, base_log_density = base(n_batch)
        phi, map_log_density = loaded_model(z)

        model_log_density = base_log_density + map_log_density
        target_log_density = target(phi)  # term from parameterisatiom goes here

        loaded_model.zero_grad()  # get rid of stored gradients
        current_loss = shifted_kl(model_log_density, target_log_density)
        current_loss.backward()  # calc gradients

        loaded_optimizer.step()
        scheduler.step(current_loss)

        if (i % 50) == 0:
            pbar.set_description(f"loss: {current_loss.item()}")
    torch.save(
        {
            "epoch": train_range[-1],
            "model_state_dict": loaded_model.state_dict(),
            "optimizer_state_dict": loaded_optimizer.state_dict(),
            "loss": current_loss,
        },
        f"{outpath}/checkpoint_{train_range[-1]}.pt",
    )
