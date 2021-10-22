# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
checkpoint.py

Module for loading neural networks and checkpoints - ensuring a copy of model
is made so that we don't get unexpected results

"""
from pathlib import Path
from glob import glob
from copy import deepcopy

import torch

from reportengine.compat import yaml


def loaded_checkpoint(checkpoint):
    """Returns a loaded checkpoint containing the state of a model."""
    if checkpoint is None:
        return None
    cp_loaded = checkpoint.load()
    return cp_loaded


def loaded_model(loaded_checkpoint, model_to_load):
    """Loads state from checkpoint if provided, returns instantiated model."""
    new_model = deepcopy(
        model_to_load
    )  # need to copy model so we don't get weird results
    if loaded_checkpoint is not None:
        new_model.load_state_dict(loaded_checkpoint["model_state_dict"])
    return new_model


def loaded_optimizer(
    loaded_model,
    loaded_checkpoint,
    optimizer,
    optimizer_params,
    scheduler,
    scheduler_params,
):
    """Loads state from checkpoint if provided, returns instantiated optimizer."""
    optim_class = getattr(torch.optim, optimizer)
    optim_instance = optim_class(loaded_model.parameters(), **optimizer_params)
    sched_class = getattr(torch.optim.lr_scheduler, scheduler)
    sched_instance = sched_class(optim_instance, **scheduler_params)
    
    # Must load optimizer *after* instantiating scheduler!
    # See https://github.com/pytorch/pytorch/issues/65342
    if loaded_checkpoint is not None:
        optim_instance.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
        sched_instance.load_state_dict(loaded_checkpoint["scheduler_state_dict"])
    return optim_instance, sched_instance


def train_range(loaded_checkpoint, epochs: int) -> tuple:
    """Returns tuple containing the indices of the next and last training iterations.

    If training from scratch, this will look like ``(0, epochs)`` where ``epochs``.
    If loading from a checkpoint, it will instead look like ``(i_cp, epochs)``
    where ``i_cp`` indexes the iteration at which the checkpoint was saved.
    """
    if loaded_checkpoint is not None:
        cp_epoch = loaded_checkpoint["epoch"]
        train_range = (cp_epoch, epochs)
    else:
        train_range = (0, epochs)
    return train_range


def current_loss(loaded_checkpoint):
    """Returns the current value of the loss function from a loaded checkpoint, or
    ``None`` if no checkpoint is provided."""
    if loaded_checkpoint is None:
        return None
    return loaded_checkpoint["loss"]


class InvalidCheckpointError(Exception):
    pass


class InvalidTrainingOutputError(Exception):
    pass


class TrainingRuncardNotFound(InvalidTrainingOutputError):
    pass


class Checkpoint:
    """Class which saves and loads checkpoints and allows checkpoints to be
    sorted"""

    def __init__(self, path: str):
        self.path = Path(path)
        try:
            self.epoch = int(self.path.stem.split("_")[-1])  # should be an int
        except ValueError:
            raise InvalidCheckpointError(
                f"{self.path} does not match expected "
                "name checkpoint: `checkpoint_<epoch>.pt`"
            )

    def __lt__(self, other):
        return self.epoch < other.epoch

    def __repr__(self):
        return str(self.path)

    def load(self):
        """Return checkpoint dictionary"""
        return torch.load(self.path)


class TrainingOutput:
    """Class which acts as container for training output, which is a directory
    containing training configuration, checkpoints and training logs
    """

    _loaded_config = None

    def __init__(self, path: str):
        self.path = Path(path)
        self.config = self.path / "runcard.yml"
        if not self.config.is_file():
            raise TrainingRuncardNotFound(
                f"Invalid training output, no runcard found at: {self.config}"
            )
        self.checkpoints = [
            Checkpoint(cp_path) for cp_path in glob(f"{self.path}/checkpoints/*")
        ]
        self.cp_ids = [cp.epoch for cp in self.checkpoints]
        self.name = self.path.name

    def get_config(self):
        if self._loaded_config is None:
            with open(self.config, "r") as f:
                self._loaded_config = yaml.safe_load(f)
        return self._loaded_config

    def as_input(self):
        inp = dict(self.get_config())  # make copy
        inp["checkpoints"] = self.checkpoints
        inp["cp_ids"] = self.cp_ids
        return inp

    def final_checkpoint(self):
        return max(self.checkpoints)
