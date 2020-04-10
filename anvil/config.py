"""
config.py

Module to parse runcards
"""
import logging

from reportengine.report import Config
from reportengine.configparser import ConfigError, element_of, explicit_node

from anvil.core import TrainingOutput
from anvil.geometry import Geometry2D
from anvil.models import real_nvp, stereographic_projection
from anvil.theories import phi_four_action, spin_hamiltonian
from anvil.distributions import normal_distribution, spherical_distribution

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

    def produce_geometry(self, lattice_length):
        return Geometry2D(lattice_length)

    def parse_theory(self, theory: str):
        return theory

    def parse_m_sq(self, m: (float, int)):
        return m

    def parse_lam(self, lam: (float, int)):
        return lam

    @explicit_node
    def produce_target(self, theory):
        """Return the function which initialises the correct action"""
        if theory == "phi_four":
            return phi_four_action
        elif theory == "spin":
            return spin_hamiltonian
        raise ConfigError(
            f"Selected theory: {theory}, has not been implemented yet",
            theory,
            ["phi_four", "spin"],
            )
    
    def parse_use_arxiv_version(self, do_use: bool):
        return do_use

    def parse_beta(self, beta: float):
        return beta

    def parse_field_dimension(self, dim: int):
        return dim

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

    @explicit_node
    def produce_base(self, base_dist: str = "normal",):
        """Return the action which loads appropriate base distribution"""
        if base_dist == "normal":
            return normal_distribution
        elif base_dist == "spherical":
            return spherical_distribution
        raise ConfigError(
            f"Base distribution: {base_dist}, has not been implemented yet",
            base_dist,
            ["normal", "spherical"],
        )

    @explicit_node
    def produce_model(self, flow_model: str = "real_nvp"):
        if flow_model == "real_nvp":
            return real_nvp
        elif flow_model == "stereographic_projection":
            return stereographic_projection
        raise ConfigError(
            f"Model: {flow_model}, has not been implemented yet",
            flow_model,
            ["real_nvp", "stereographic_projection"],
        )

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

    def produce_training_geometry(self, training_output):
        with self.set_context(ns=self._curr_ns.new_child(training_output.as_input())):
            _, geometry = self.parse_from_(None, "geometry", write=False)
        return geometry

    def parse_optimizer(self, optim: str):
        valid_optimizers = ("adam", "adadelta")
        if optim not in valid_optimizers:
            raise ConfigError(
                f"Invalid optimizer choice: {optim}", optim, valid_optimizers
            )
        return optim

    def parse_optimizer_kwargs(self, kwargs: dict, optimizer):
        # This will only be executed if optimizer is defined in the runcard
        if optimizer == "adam":
            valid_kwargs = ("lr", "lr_decay", "weight_decay", "eps")
        if optimizer == "adadelta":
            valid_kwargs = ("lr", "rho", "weight_decay", "eps")

        if not all([arg in valid_kwargs for arg in kwargs]):
            raise ConfigError(
                f"Valid optimizer_kwargs for {optimizer} are {', '.join([arg for arg in valid_kwargs])}"
            )
        return kwargs

    def parse_scheduler_kwargs(self, kwargs: dict):
        if "patience" not in kwargs:
            kwargs["patience"] = 500  # problem setting default in config parser?
        return kwargs

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
