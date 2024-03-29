# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
train.py

Module containing functions required to train model
"""
import logging
import signal
from tqdm import tqdm

import torch

from anvil.utils import get_num_parameters, handler

log = logging.getLogger(__name__)


signal.signal(signal.SIGTERM, handler)  # termination
signal.signal(signal.SIGINT, handler)  # keyboard interrupt


def save_checkpoint(
    outpath: str, epoch: int, loss: float, model, optimizer, scheduler
) -> None:
    """Saves a the state of the model, optimizer and scheduler as a checkpoint file
    for later reloading."""
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        f"{outpath}/checkpoints/checkpoint_{epoch}.pt",
    )
    tqdm.write(f"Checkpoint saved at epoch {epoch}")


def reverse_kl(
    model_log_density: torch.Tensor, target_log_density: torch.Tensor
) -> torch.Tensor:
    r"""Estimate of the reverse Kullbach-Leibler divergence between the model
    and the target density, obtained by averaging over a batch of configurations
    generated by the model.

    Parameters
    ----------
    model_log_density
        column of the logarithm of the probability density for the batch of
        configurations generated by the model, as given by the change of
        variables formula.
    target_log_density
        column containing the logarithm of the probability density for the
        batch of configurations, defined by the action of the field theory.

    Returns
    -------
    torch.Tensor
        torch tensor with single element, corresponding to the estimate of the
        reverse KL divergence.

    Notes
    -----
    Only terms which depend on the parameters of the model are necessary for
    optimisation. Hence, model_log_density could be simply the logarithm of the
    Jacobian determinant of the learnable layers in the normalising flow, and
    target_log_density need only be the negative action.
    """
    return torch.mean(model_log_density - target_log_density, dim=0)


def training_update(
    loaded_model,
    base_dist,
    target_dist,
    n_batch: int,
    current_loss: float,
    optimizer,
    scheduler,
) -> float:
    """A single training update or 'epoch'.

    Parameters
    ----------
    loaded_model
        Flow model whose parameters are to be updated.
    base_dist
        Distribution object that generates latent variables to be passed
        through the flow model.
    target_dist
        Distribution or field theory which would like to use a trained model
        to sample from.
    n_batch
        Number of configurations in a 'batch', i.e. which are used to estimate
        the gradient of the objective function and hence update the model
        parameters.
    current_loss
        The current value of the loss or objective function.
    optimizer
        Optimization algorithm which will be used to update the model parameters.
    scheduler
        Learning rate scheduler.

    Returns
    -------
    float
        The value of the loss or objective function after the training update.
    """
    # Generate latent variables
    z, base_log_density = base_dist(n_batch)

    # Transform latents -> candidate configurations (gradients tracked)
    # NOTE: base_log_density not strictly necessary - could pass 0
    phi, model_log_density = loaded_model(z, base_log_density)

    # Compute objective function (gradients tracked)
    target_log_density = target_dist.log_density(phi)
    current_loss = reverse_kl(model_log_density, target_log_density)

    # Backprop gradients and update model parameters
    optimizer.zero_grad()  # zero gradients from prev minibatch
    current_loss.backward()  # accumulate new gradients
    optimizer.step()  # update model parameters

    # TODO: we have different scheduler updates depending on which we're using,
    # which are not currently accounted for. E.g. would require
    # scheduler.step(current_loss) for ReduceLROnPlateau
    scheduler.step()  # update learning rate

    return current_loss


def train(
    loaded_model,
    base_dist,
    target_dist,
    *,
    train_range: tuple,
    n_batch: int,
    outpath: str,
    current_loss: float,
    loaded_optimizer: tuple,
    save_interval: int = 1000,
    loss_sample_interval: int = 25,
):
    """Loop over training updates, periodically saving checkpoints.

    Repeatedly calls :py:func:`anvil.train.training_update` and
    :py:func:`anvil.train.save_checkpoint` , until a prescribed number of training
    updates have been performed.

    Parameters
    ----------
    loaded_model
        Flow model whose parameters are to be updated.
    base_dist
        Distribution object that generates latent variables to be passed
        through the flow model.
    target_dist
        Distribution or field theory which would like to use a trained model
        to sample from.
    train_range
        Tuple containing indices of the next and last training iteration.
    n_batch
        Number of configurations in a 'batch', i.e. which are used to estimate
        the gradient of the objective function and hence update the model
        parameters.
    outpath
        Path to directory where training outputs are to be saved.
    current_loss
        The current value of the loss or objective function.
    loaded_optimizer
        Tuple containing loaded optimizer and scheduler.
    save_interval
        Number of training updates between checkpoints.
    loss_sample_interval
        Rate at which the loss or objective function is sampled for post-analysis.
        This also controls the rate at which the progress bar is updated.

    Returns
    -------
    loaded_model

    """

    # Let the user know the total number of model parameters
    # TODO: should have a --verbose option which controls whether we care about stuff like this
    num_parameters = get_num_parameters(loaded_model)
    log.info(f"Model has {num_parameters} trainable parameters.")

    optimizer, scheduler = loaded_optimizer

    # TODO: should provide option to not use tqdm progress bar, e.g. when
    # running benchmark test with pytest, github CI
    current_epoch, final_epoch = train_range
    pbar = tqdm(total=(final_epoch - current_epoch))

    # Stores samples of (epoch, loss)
    history = []

    # try, finally statement allows us to save checkpoint in case of unexpected
    # SIGTERM or SIGINT
    try:
        while current_epoch < final_epoch:

            if (current_epoch % save_interval) == 0:
                save_checkpoint(
                    outpath,
                    epoch=current_epoch,
                    loss=current_loss,
                    model=loaded_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

            current_loss = training_update(
                loaded_model,
                base_dist,
                target_dist,
                n_batch,
                current_loss,
                optimizer,
                scheduler,
            )
            # Increment counter immediately after training update
            current_epoch += 1
            pbar.update()

            if (current_epoch % loss_sample_interval) == 0:
                loss = float(current_loss.item())
                pbar.set_description(f"loss: {loss}")
                history.append(f"{current_epoch} {loss}\n")

    finally:  # executed after while loop, or if sys.exit is called
        pbar.close()
        save_checkpoint(
            outpath,
            epoch=current_epoch,
            loss=current_loss,
            model=loaded_model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        with open(f"{outpath}/loss.txt", "a") as f:
            f.writelines(history)

    return loaded_model
