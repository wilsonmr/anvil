"""
config.py

Module to parse runcards
"""
import logging

from reportengine.report import Config
from reportengine.configparser import ConfigError, element_of, explicit_node

from anvil.core import TrainingOutput
from anvil.train import OPTIMIZER_OPTIONS, reduce_lr_on_plateau
from anvil.models import RealNVP
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
        return target

    def parse_base(self, base: str):
        return base

    def parse_mean(self, mean: (float, int)):
        return mean

    def parse_sigma(self, sigma: (float, int)):
        return sigma

    def parse_support(self, supp: list):
        return supp

    def parse_concentration(self, conc: float):
        return conc

    def parse_m_sq(self, m: (float, int)):
        return m

    def parse_lam(self, lam: (float, int)):
        return lam

    def parse_use_arxiv_version(self, do_use: bool):
        return do_use

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

    def parse_hidden_nodes(self, hid_spec):
        return hid_spec

    def produce_network_kwargs(self, hidden_nodes):
        """Returns a dictionary that is the necessary kwargs for the NVP class
        This means in the future if we change the class to have more flexibility
        with regard to network spec then we can use this function to bridge
        backwards compatibility"""
        hidden_nodes = tuple(hidden_nodes)
        return dict(affine_hidden_shape=hidden_nodes)

    def parse_n_affine(self, n: int):
        return n

    def parse_n_batch(self, nb: int):
        return nb

    def produce_model(self, lattice_size, n_affine, network_kwargs):
        model = RealNVP(size_in=lattice_size, n_affine=n_affine, **network_kwargs)
        return model

    def parse_epochs(self, epochs: int):
        return epochs

    def parse_save_interval(self, save_int: int):
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
        with self.set_context(ns=self._curr_ns.new_child(training_context)):
            _, geometry = self.parse_from_(None, "geometry", write=False)
        return geometry

    @explicit_node
    def produce_loaded_optimizer(self, optimizer):
        try:
            return OPTIMIZER_OPTIONS[optimizer]
        except KeyError:
            raise ConfigError(
                f"Invalid optimizer {optimizer}", optimizer, OPTIMIZER_OPTIONS.keys()
            )

    @explicit_node
    def produce_scheduler(self):
        return reduce_lr_on_plateau

    def parse_target_length(self, targ: int):
        return targ

    def parse_thermalisation(self, therm: (int, type(None))):
        if therm is None:
            log.warning("Not Performing thermalisation")
            return therm
        if therm < 1:
            raise ConfigError(
                "thermalisation must be greater than or equal to 1 or be None"
            )
        return therm

    def parse_sample_interval(self, interval: (int, type(None))):
        if interval is None:
            return interval
        if interval < 1:
            raise ConfigError("sample_interval must be greater than or equal to 1")
        log.warning(f"Using user specified sample_interval: {interval}")
        return interval

    def parse_n_boot(self, n_boot: int):
        if n_boot < 2:
            raise ConfigError("n_boot must be greater than 1")
        log.warning(f"Using user specified n_boot: {n_boot}")
        return n_boot

    @element_of("windows")
    def parse_window(self, window: float):
        if window < 0:
            raise ConfigError("window must be positive")
        log.warning(f"Using user specified window 'S': {window}")
        return window
