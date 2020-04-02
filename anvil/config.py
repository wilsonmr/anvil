"""
config.py

Module to parse runcards
"""
import logging

from reportengine.report import Config
from reportengine.configparser import ConfigError, element_of

from anvil.core import PhiFourAction, TrainingOutput
from anvil.models import RealNVP
from anvil.geometry import Geometry2D
from anvil.distributions import NormalDist

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

    def parse_m_sq(self, m: (float, int)):
        return m

    def parse_lam(self, lam: (float, int)):
        return lam

    def parse_use_arxiv_version(self, do_use: bool):
        return do_use

    def produce_action(self, m_sq, lam, geometry, use_arxiv_version):
        return PhiFourAction(
            m_sq, lam, geometry=geometry, use_arxiv_version=use_arxiv_version
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

    def produce_generator(
        self,
        lattice_size: int,
        base_dist: str = "normal",
        field_dimension: int = 1,
    ):
        if base_dist == "normal":
            return NormalDist(
                lattice_volume=lattice_size,
                field_dimension=field_dimension,
            )
        else:
            raise NotImplementedError

    def produce_model(self, generator, n_affine, network_kwargs):
        model = RealNVP(generator=generator, n_affine=n_affine, **network_kwargs)
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
        with self.set_context(ns=self._curr_ns.new_child(training_output.as_input())):
            _, geometry = self.parse_from_(None, "geometry", write=False)
            _, model = self.parse_from_(None, "model", write=False)
            _, action = self.parse_from_(None, "action", write=False)
            _, cps = self.parse_from_(None, "checkpoints", write=False)

        return dict(geometry=geometry, model=model, action=action, checkpoints=cps)

    def produce_training_geometry(self, training_output):
        with self.set_context(ns=self._curr_ns.new_child(training_output.as_input())):
            _, geometry = self.parse_from_(None, "geometry", write=False)
        return geometry

    def parse_optimizer(self, optim: str):
        valid_optimizers = ("adam", "adadelta")
        if optim not in valid_optimizers:
            raise ConfigError(
                f"optimizer must be one of {', '.join([opt for opt in valid_optimizers])}"
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
        if n_samples < 2:
            raise ConfigError("n_boot must be greater than 1")
        log.warning(f"Using user specified n_boot: {n_samples}")
        return n_samples

    @element_of("windows")
    def parse_window(self, window: float):
        if window < 0:
            raise ConfigError("window must be positive")
        log.warning(f"Using user specified window 'S': {window}")
        return window
