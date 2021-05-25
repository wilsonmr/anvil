# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
config.py

Module to parse runcards
"""
from random import randint
from sys import maxsize
import logging
import platform

from reportengine.report import Config
from reportengine.configparser import ConfigError, element_of, explicit_node

from anvil.geometry import Geometry2D
from anvil.checkpoint import TrainingOutput
from anvil.models import LAYER_OPTIONS
from anvil.distributions import BASE_OPTIONS, TARGET_OPTIONS

log = logging.getLogger(__name__)


class ConfigParser(Config):
    """Extend the reportengine Config class for anvil-specific
    objects
    """

    def parse_lattice_length(self, length: int) -> int:
        """The number of nodes along each spatial dimension."""
        return length

    def parse_lattice_dimension(self, dim: int) -> int:
        """The number of spatial dimensions."""
        if dim != 2:
            raise ConfigError("Currently only 2 dimensions is supported")
        return dim

    def produce_lattice_size(self, lattice_length: int, lattice_dimension: int) -> int:
        """The total number of nodes on the lattice."""
        return pow(lattice_length, lattice_dimension)

    def produce_size_half(self, lattice_size: int) -> int:
        """Half of the number of nodes on the lattice.

        This defines the size of the input layer to the neural networks.
        """
        # NOTE: we may want to make this more flexible
        if (lattice_size % 2) != 0:
            raise ConfigError("Lattice size is expected to be an even number")
        return int(lattice_size / 2)

    def produce_geometry(self, lattice_length: int):
        """Returns the geometry object defining the lattice."""
        return Geometry2D(lattice_length)

    @explicit_node
    def produce_target_dist(self, target: str):
        """Returns the function which initialises the correct action"""
        try:
            return TARGET_OPTIONS[target]
        except KeyError:
            raise ConfigError(
                f"invalid target distribution {target}", target, TARGET_OPTIONS.keys()
            )

    @explicit_node
    def produce_base_dist(self, base: str):
        """Returns the action which loads appropriate base distribution"""
        try:
            return BASE_OPTIONS[base]
        except KeyError:
            raise ConfigError(
                f"Invalid base distribution {base}", base, BASE_OPTIONS.keys()
            )

    def parse_sigma(self, sigma: float) -> float:
        """The standard deviation of a normal distribution."""
        return sigma

    def parse_couplings(self, couplings: dict) -> dict:
        """A dict containing the couplings for the target field theory."""
        return couplings  # TODO: obviously need to be more fool-proof about this

    def parse_parameterisation(self, param: str) -> str:
        """A string defining the parameterisation used for the target theory."""
        return param

    @explicit_node
    def produce_layer_action(self, layer: str):
        """Given a string, returns the flow model action indexed by that string."""
        try:
            return LAYER_OPTIONS[layer]
        except KeyError:
            raise ConfigError(f"Invalid model {layer}", layer, LAYER_OPTIONS.keys())

    def parse_n_batch(self, nb: int) -> int:
        """Batch size for training."""
        return nb

    def parse_epochs(self, epochs: int) -> int:
        """Number of training iterations, i.e. updates of the model parameters."""
        return epochs

    def parse_save_interval(self, save_int: int) -> int:
        """A checkpoint containing the model state will be written every ``save_interval``
        training iterations."""
        return save_int

    @element_of("training_outputs")
    def parse_training_output(self, path: str):
        """Given a path to a training directory, returns an object that interfaces with
        this directory."""
        return TrainingOutput(path)

    @element_of("cp_ids")
    def parse_cp_id(self, cp: (int, type(None))) -> (int, None):
        return cp

    @element_of("checkpoints")
    def produce_checkpoint(
        self,
        cp_id: (int, type(None)),
        training_output,
    ):
        """Attempts to return a checkpoint object extracted from a training output.

        - If ``cp_id == None``, no checkpoint is returned.
        - If ``cp_id == -1``, the checkpoint with the highest ``cp_id`` is returned.
        - Otherwise, attempts to load checkpoint with id ``cp_id``.
        """
        if cp_id is None:
            return None
        if cp_id == -1:
            return training_output.final_checkpoint()
        if cp_id not in training_output.cp_ids:
            raise ConfigError(f"Checkpoint {cp_id} not found in {training_output.path}")
        # get index from training_output class
        return training_output.checkpoints[training_output.cp_ids.index(cp_id)]

    def produce_training_context(self, training_output) -> dict:
        """Given a training output, produces the context of that training as a dict."""
        # NOTE: This seems a bit hacky, exposing the entire training configuration
        # file - hopefully doesn't cause any issues..
        return training_output.as_input()

    def produce_training_geometry(self, training_context: dict):
        """Produces the geometry object used in training."""
        with self.set_context(ns=self._curr_ns.new_child(training_context)):
            _, geometry = self.parse_from_(None, "geometry", write=False)
        return geometry

    def parse_optimizer(self, optimizer: str) -> str:
        """A label for the optimization algorithm to use during training.

        An optimizer is loaded using ``getattr(torch.optim, <optimizer>)``. Therefore
        this label must correspond to a `valid PyTorch optimizer`_.

        .. _valid PyTorch optimizer: https://pytorch.org/docs/stable/optim.html#algorithms
        """
        return optimizer

    def parse_optimizer_params(self, params: dict) -> dict:
        """Parameters for the optimization algorithm.

        Consult the documentation for `valid PyTorch optimizer`_ s.
        """
        return params

    def parse_scheduler(self, scheduler: str) -> str:
        """A label for the learning rate scheduler to use during training.

        An scheduler is loaded using ``getattr(torch.optim.lr_scheduler, <optimizer>)``.
        Therefore this label must correspond a `valid PyTorch scheduler`_.

        .. _valid PyTorch scheduler: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        """
        return scheduler

    def parse_scheduler_params(self, params: dict) -> dict:
        """Parameters for the learning rate scheduler.

        Consult the documentation for `valid PyTorch scheduler`_ s.
        """
        return params

    def parse_sample_size(self, size: int) -> int:
        """The number of configurations in the output sample."""
        return size

    def parse_thermalization(self, therm: (int, type(None))) -> (int, type(None)):
        """A number of Markov chain steps to be discarded before beginning to select
        configurations for the output sample."""
        if therm is None:
            log.warning("Not Performing thermalization")
            return None
        if therm < 1:
            raise ConfigError(
                "Thermalization must be greater than or equal to 1 or be None"
            )
        return therm

    def parse_sample_interval(self, interval: (int, type(None))) -> (int, type(None)):
        """A number of Markov chain steps to discard between configurations that are
        selected for the output sample.

        Can be specified by the user in the runcard, or left to an automatic
        calculation based on the acceptance rate of the Metropolis-Hastings algorith.
        """
        if interval is None:
            log.info("No sample_interval provided - will be calculated 'on the fly'.")
            return None
        if interval < 1:
            raise ConfigError("sample_interval must be greater than or equal to 1")
        log.info(f"Using user specified sample_interval: {interval}")
        return interval

    def parse_bootstrap_sample_size(self, n_boot: int) -> int:
        """The size of the bootstrap sample."""
        if n_boot < 2:
            # TODO: would be nice to have the option to perform analysis without bootstrapping
            raise ConfigError("bootstrap sample size must be greater than 1")
        log.warning(f"Using user specified bootstrap sample size: {n_boot}")
        return n_boot

    def produce_bootstrap_seed(self, manual_bootstrap_seed: (int, None) = None) -> int:
        """Optional seed for the random number generator which generates the bootstrap
        sample, for the purpose of reproducibility."""
        if manual_bootstrap_seed is None:
            return randint(0, maxsize)
        # numpy is actually this strict but let's keep it sensible.
        if (manual_bootstrap_seed < 0) or (manual_bootstrap_seed > 2 ** 32):
            raise ConfigError("Seed is outside of appropriate range: [0, 2 ** 32]")
        return manual_bootstrap_seed

    @element_of("windows")
    def parse_window(self, window: float) -> float:
        """A numerical factor featuring in the calculation of the optimal 'window'
        size, which is then used to measure the integrated autocorrelation time of
        observables.

        Suggested values are between 1 and 2. However, this should be judged by
        checking that the integrated autocorrelation has approximately plateaued
        at the optimal window size.

        See :py:func:`anvil.observables.automatic_windowing_function`.
        """
        if window < 0:
            raise ConfigError("window must be positive")
        log.warning(f"Using user specified window 'S' parameter: {window}")
        return window

    def produce_use_multiprocessing(self) -> bool:
        """Don't use Python multiprocessing on MacOS"""
        if platform.system() == "Darwin":
            return False
        return True
