"""
train.py

Module containing functions required to train model
"""

from tqdm import tqdm

import torch
import torch.optim as optim

from math import pi

def shifted_kl(log_tilde_p: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    r"""Sample mean of the shifted Kullbach-Leibler divergence between target
    and model distribution.

    Parameters
    ----------
    log_tilde_p: torch.Tensor
        column of log (\tilde p) for a sample of states, which is returned by
        forward pass of `RealNVP` model
    action: torch.Tensor
        column of actions S(\phi) for set of sample states

    Returns
    -------
    out: torch.Tensor
        torch tensor with single element, corresponding to the estimation of the
        shifted K-L for set of sample states.

    """
    return torch.mean(log_tilde_p + action, dim=0)


def train(
    loaded_model,
    action,
    *,
    train_range,
    save_interval,
    n_batch=2000,
    outpath,
    current_loss,
    loaded_optimizer,
):
    """training loop of model"""
    # create your optimizer and a scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(loaded_optimizer, patience=500)
    # let's use tqdm to see progress
    pbar = tqdm(range(*train_range), desc=f"loss: {current_loss}")
    n_units = loaded_model.size_in
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
        z = loaded_model.generator(n_batch)
        phi = loaded_model.inverse_map(z)
        target = action(phi)
        output = loaded_model(phi)

        loaded_model.zero_grad()  # get rid of stored gradients
        current_loss = shifted_kl(output, target)
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
