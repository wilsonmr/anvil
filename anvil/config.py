"""
config.py

Module to parse runcards
"""
import logging

from reportengine.report import Config
from reportengine.configparser import ConfigError, element_of, explicit_node

from anvil.core import TrainingOutput
from anvil.train import OPTIMIZER_OPTIONS, reduce_lr_on_plateau
from anvil.models import MODEL_OPTIONS
from anvil.geometry import Geometry2D
from anvil.distributions import BASE_OPTIONS, TARGET_OPTIONS

log = logging.getLogger(__name__)


class ConfigParser(Config):
    """Extend the reportengine Config class for anvil-specific
    objects
    """

    def parse_lattice_length(self, length: int):
        return length

    def parse_lattice_dimension(self, dim: int):
        """Parse lattice dimension from runcard"""
        if dim != 2:
            raise ConfigError("Currently only 2 dimensions is supported")
        return dim

    def produce_lattice_size(self, lattice_length, lattice_dimension):
        """returns the total number of nodes on lattice"""
        return pow(lattice_length, lattice_dimension)

    def produce_config_size(self, lattice_size, target_dimension=1):
        return target_dimension * lattice_size

    def produce_geometry(self, lattice_length):
        return Geometry2D(lattice_length)

    def parse_target(self, target: str):
        """String specifying target distrbution."""
        return target

    def parse_base(self, base: str):
        """String specifying base distribution."""
        return base

    @explicit_node
    def produce_target_dist(self, target: str):
        """Return the function which initialises the correct action"""
        try:
            return TARGET_OPTIONS[target]
        except KeyError:
            raise ConfigError(
                f"invalid target distribution {target}", target, TARGET_OPTIONS.keys()
            )

    @explicit_node
    def produce_base_dist(self, base: str):
        """Return the action which loads appropriate base distribution"""
        try:
            return BASE_OPTIONS[base]
        except KeyError:
            raise ConfigError(
                f"Invalid base distribution {base}", base, BASE_OPTIONS.keys()
            )

    def parse_mean(self, mean: (float, int)):
        """Mean of normal or von Mises distribution."""
        return mean

    def parse_sigma(self, sigma: (float, int)):
        """Standard deviation of normal distribution."""
        return sigma

    def parse_support(self, supp: list):
        """Support of uniform distrbution."""
        return supp

    def parse_concentration(self, conc: float):
        """Concentration parameter of von Mises distribution."""
        return conc

    def parse_m_sq(self, m: (float, int)):
        """Bare mass squared in scalar theory."""
        return m

    def parse_lam(self, lam: (float, int)):
        """Coefficient of quartic interaction in phi^4 theory."""
        return lam

    def parse_use_arxiv_version(self, do_use: bool):
        """If true, use the conventional phi^4 action. If false,
        there is an additional factor of 1/2 for the kinetic part
        of the phi^4 action."""
        return do_use

    def parse_network_spec(self, net_spec: dict):
        """A dictionary where each element is itself a dictionary containing the
        parameters required to construct a neural network."""
        required_networks = ("s", "t")  # NOTE: real_nvp. Will generalise in future PR
        for k in required_networks:
            if k not in net_spec:
                raise ConfigError(
                    f"network_spec should contain keys {required_networks} but contains keys {net_spec.keys()}"
                )
            if not isinstance(net_spec[k], dict):
                raise ConfigError(
                    f"network spec does not contain a dictionary associated to key {k}"
                )
        return net_spec

    def parse_standardise_inputs(self, do_stand: bool):
        """Flag specifying whether to standardise input vectors before
        passing them through a neural network."""
        return do_stand

    @explicit_node
    def produce_target_dist(self, target):
        """Return the function which initialises the correct action"""
        try:
            return TARGET_OPTIONS[target]
        except KeyError:
            raise ConfigError(
                f"invalid target distribution {target}", target, TARGET_OPTIONS.keys()
            )

    @explicit_node
    def produce_base_dist(
        self, base,
    ):
        """Return the action which loads appropriate base distribution"""
        try:
            return BASE_OPTIONS[base]
        except KeyError:
            raise ConfigError(
                f"Invalid base distribution {base}", base, BASE_OPTIONS.keys()
            )

    def parse_n_affine(self, n: int):
        """Number of affine layers."""
        return n

    def parse_n_batch(self, nb: int):
        """Batch size for training."""
        return nb

    def parse_model(self, model: str):
        """Label for normalising flow model."""
        return model

    @explicit_node
    def produce_flow_model(self, model):
        """Return the action which instantiates the normalising flow model."""
        try:
            return MODEL_OPTIONS[model]
        except KeyError:
            raise ConfigError(
                f"invalid flow model {model}", model, MODEL_OPTIONS.keys()
            )

    def parse_epochs(self, epochs: int):
        """Number of epochs to train. Equivalent to number of passes
        multiplied by the batch size."""
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

    def produce_training_geometry(self, training_context):
        """Produces the geometry object used in training."""
        with self.set_context(ns=self._curr_ns.new_child(training_context)):
            _, geometry = self.parse_from_(None, "geometry", write=False)
        return geometry

    @explicit_node
    def produce_loaded_optimizer(self, optimizer):
        """Returns an action which itself returns a torch.optim.Optimizer object
        that knows about the current state of the model."""
        try:
            return OPTIMIZER_OPTIONS[optimizer]
        except KeyError:
            raise ConfigError(
                f"Invalid optimizer {optimizer}", optimizer, OPTIMIZER_OPTIONS.keys()
            )

    @explicit_node
    def produce_scheduler(self):
        """Currently fixed to ReduceLROnPlateau"""
        return reduce_lr_on_plateau

    def parse_target_length(self, targ: int):
        """Target number of decorrelated field configurations to generate."""
        return targ

    def parse_thermalisation(self, therm: (int, type(None))):
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

    def parse_n_boot(self, n_boot: int):
        """Size of the bootstrap sample."""
        if n_boot < 2:
            raise ConfigError("n_boot must be greater than 1")
        log.warning(f"Using user specified n_boot: {n_boot}")
        return n_boot

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
