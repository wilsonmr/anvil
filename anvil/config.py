"""
config.py

Module to parse runcards
"""
import logging
import torch.optim
from random import randint
from sys import maxsize

from reportengine.report import Config
from reportengine.configparser import ConfigError, element_of, explicit_node

from anvil.geometry import Geometry2D
from anvil.checkpoint import TrainingOutput
from anvil.models import MODEL_OPTIONS
from anvil.distributions import BASE_OPTIONS, THEORY_OPTIONS

log = logging.getLogger(__name__)


class ConfigParser(Config):
    """Extend the reportengine Config class for anvil-specific
    objects
    """

    def parse_lattice_length(self, length: int):
        return length

    def produce_lattice_size(self, lattice_length, lattice_dimension):
        """returns the total number of nodes on lattice"""
        return lattice_length ** 2

    def produce_size_half(self, lattice_size):
        """Given the number of nodes in a field configuration, return an integer
        of lattice_size/2 which is the size of the input vector for each coupling layer.
        """
        # NOTE: we may want to make this more flexible
        if (lattice_size % 2) != 0:
            raise ConfigError("Lattice size is expected to be an even number")
        return int(lattice_size / 2)

    def produce_geometry(self, lattice_length):
        return Geometry2D(lattice_length)

    def produce_target(self, theory, geometry, parameterisation, couplings):
        theory_class = THEORY_OPTIONS[theory]
        constructor = getattr(theory_class, f"from_{parameterisation}")
        instance = constructor(geometry, **couplings)
        return instance

    def produce_base(self, latents, latent_params, lattice_size):
        base_class = BASE_OPTIONS[latents]
        instance = base_class(lattice_size, **latent_params)
        return instance

    def produce_model_to_load(self, model, model_params, size_half):
        model_class = MODEL_OPTIONS[model]
        instance = model_class(size_half, **model_params)
        return instance

    def parse_n_batch(self, nb: int):
        """Batch size for training."""
        return nb

    def parse_epochs(self, epochs: int):
        """Number of optimization steps during training."""
        return epochs

    def parse_save_interval(self, save_int: int):
        """Interval at which the model state is saved, in units of epochs."""
        return save_int

    @element_of("training_outputs")
    def parse_training_output(self, path: str):
        return TrainingOutput(path)

    @element_of("cp_ids")
    def parse_cp_id(self, cp: (int, type(None))):
        return cp

    @element_of("checkpoints")
    def produce_checkpoint(self, cp_id=None, training_output=None):
        if cp_id is None:
            return None
        if cp_id == -1:
            return training_output.final_checkpoint()
        if cp_id not in training_output.cp_ids:
            raise ConfigError(f"Checkpoint {cp_id} not found in {training_output.path}")
        # get index from training_output class
        return training_output.checkpoints[training_output.cp_ids.index(cp_id)]

    def produce_training_context(self, training_output):
        """Given a training output produce the context of that training"""
        # NOTE: This seems a bit hacky, exposing the entire training configuration
        # file - hopefully doesn't cause any issues..
        return training_output.as_input()

    def produce_geometry_from_training(self, training_context):
        """Produces the geometry object used in training."""
        with self.set_context(ns=self._curr_ns.new_child(training_context)):
            _, geometry = self.parse_from_(None, "geometry", write=False)
        return geometry

    def produce_base_from_training(self, training_context):
        with self.set_context(ns=self._curr_ns.new_child(training_context)):
            _, base = self.parse_from_(None, "base", write=False)
        return base
    
    def produce_target_from_training(self, training_context):
        with self.set_context(ns=self._curr_ns.new_child(training_context)):
            _, target = self.parse_from_(None, "target", write=False)
        return target
    
    def produce_model_to_load_from_training(self, training_context):
        with self.set_context(ns=self._curr_ns.new_child(training_context)):
            _, model_to_load = self.parse_from_(None, "model_to_load", write=False)
        return model_to_load

    def parse_optimizer(self, optimizer):
        # NOTE: requires loaded_model for instance
        try:
            optim_class = getattr(torch.optim, optimizer)  # could make case-insensitive
        except KeyError:
            raise ConfigError(f"Invalid optimizer {optimizer}. Consult torch.optim.")
        return optim_class

    def parse_optimizer_params(self, params, optimizer):
        try:
            test = optimizer(
                [{"params": []}],
                **params,
            )
        except TypeError as error:
            print(error)
            raise ConfigError(
                f"Invalid optimizer keyword argument dict. Consult documentation for {optimizer}."
            )
        return params

    def parse_scheduler(self, scheduler):
        try:
            sched_class = getattr(
                torch.optim.lr_scheduler, scheduler
            )  # could make case-insensitive
        except KeyError:
            raise ConfigError(f"Invalid scheduler {scheduler}. Consult torch.optim.")
        return sched_class

    def parse_scheduler_params(self, params, scheduler):
        try:
            test = scheduler(
                torch.optim.Optimizer([{"params": [], "lr": 1}], {}),
                **params,
            )
        except TypeError as error:
            print(error)
            raise ConfigError(
                f"Invalid scheduler keyword argument dict. Consult documentation for {scheduler}"
            )
        return params

    def parse_sample_size(self, size: int):
        """Number of field configurations in output sample."""
        return size

    def parse_thermalization(self, therm: (int, type(None))):
        """Number of Markov chain steps to discard to allow the chain to
        reach an approximately stationary distribution."""
        if therm is None:
            log.warning("Not Performing thermalisation")
            return therm
        if therm < 1:
            raise ConfigError(
                "thermalisation must be greater than or equal to 1 or be None"
            )
        return therm

    def parse_sample_interval(self, interval: (int, type(None))):
        """Number of Markov chain steps to discard between appending configurations
        to the sample. Should be large enough so that configurations have become
        decorrelated.

        Can be specified by the user in the runcard, or left to an automatic
        calculation based on the acceptance rate of the Metropolis-Hastings algorith.
        """
        if interval is None:
            return interval
        if interval < 1:
            raise ConfigError("sample_interval must be greater than or equal to 1")
        log.warning(f"Using user specified sample_interval: {interval}")
        return interval

    def parse_bootstrap_sample_size(self, n_boot: int):
        """Size of the bootstrap sample."""
        if n_boot < 2:
            raise ConfigError("bootstrap_sample_size must be greater than 1")
        log.warning(f"Using user specified bootstrap sample size: {n_boot}")
        return n_boot

    def produce_bootstrap_seed(self):
        return randint(0, maxsize)

    @element_of("windows")
    def parse_window(self, window: float):
        """A numerical factor featuring in the calculation of the optimal 'window'
        size, which is then used to measure the integrated autocorrelation time of
        observables.

        Suggested values are between 1 and 2. However, this should be judged by
        checking that the integrated autocorrelation has approximately plateaued
        at the optimal window size.

        See `automatic_windowing_function` in the observables module for more details.
        """
        if window < 0:
            raise ConfigError("window must be positive")
        log.warning(f"Using user specified window 'S': {window}")
        return window
