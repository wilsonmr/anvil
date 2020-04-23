"""
train.py

Module containing functions required to train model
"""

from tqdm import tqdm

import torch
import torch.optim as optim


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
    base_dist,
    target_dist,
    *,
    train_range,
    save_interval,
    n_batch=2000,
    outpath,
    current_loss,
    loaded_optimizer,
    scheduler,
):
    """training loop of model"""
    # let's use tqdm to see progress
    pbar = tqdm(range(*train_range), desc=f"loss: {current_loss}")
    n_units = base_dist.size_out
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
        z, base_log_density = base_dist(n_batch)
        phi, map_log_density = loaded_model(z)

        model_log_density = base_log_density + map_log_density
        target_log_density = target_dist.log_density(phi)

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


def adam(
    loaded_model,
    loaded_checkpoint,
    *,
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0,
    amsgrad=False,
):
    optimizer = optim.Adam(
        loaded_model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
    )
    if loaded_checkpoint is not None:
        optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
    return optimizer


def adadelta(
    loaded_model, loaded_checkpoint, *, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0
):
    optimizer = optim.Adadelta(
        loaded_model.parameters(), lr=lr, rho=rho, eps=eps, weight_decay=weight_decay
    )
    if loaded_checkpoint is not None:
        optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
    return optimizer


def stochastic_gradient_descent(
    loaded_model,
    loaded_checkpoint,
    *,
    lr,
    momentum=0,
    dampening=0,
    weight_decay=0,
    nesterov=False,
):
    optimizer = optim.SGD(
        loaded_model.parameters(),
        lr=lr,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )
    if loaded_checkpoint is not None:
        optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
    return optimizer


def rms_prop(
    loaded_model,
    loaded_checkpoint,
    *,
    lr=0.01,
    alpha=0.99,
    eps=1e-08,
    weight_decay=0,
    momentum=0,
    centered=False,
):
    optimizer = optim.RMSprop(
        loaded_model.parameters(),
        lr=lr,
        alpha=alpha,
        eps=eps,
        weight_decay=weight_decay,
        momentum=momentum,
        centered=centered,
    )
    if loaded_checkpoint is not None:
        optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
    return optimizer


def reduce_lr_on_plateau(
    loaded_optimizer,
    *,
    mode="min",
    factor=0.1,
    patience=500,  # not the PyTorch default
    verbose=False,
    threshold=0.0001,
    threshold_mode="rel",
    cooldown=0,
    min_lr=0,
    eps=1e-08,
):
    return optim.lr_scheduler.ReduceLROnPlateau(
        loaded_optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        verbose=verbose,
        threshold=threshold,
        threshold_mode=threshold_mode,
        cooldown=cooldown,
        min_lr=min_lr,
        eps=eps,
    )


OPTIMIZER_OPTIONS = {
    "adam": adam,
    "adadelta": adadelta,
    "sgd": stochastic_gradient_descent,
    "rms_prop": rms_prop,
}
