"""
config.py

Module to parse runcards
"""
import logging

from reportengine.report import Config
from reportengine.configparser import ConfigError, element_of, explicit_node

from anvil.core import TrainingOutput
from anvil.geometry import Geometry2D
from anvil.models import real_nvp, project_circle, project_sphere
from anvil.theories import phi_four_action, xy_hamiltonian, heisenberg_hamiltonian
from anvil.distributions import (
    normal_distribution,
    uniform_distribution,
    circular_uniform_distribution,
    spherical_uniform_distribution,
)
from anvil.fields import scalar_field, xy_field, heisenberg_field

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

    def parse_theory(self, theory: str):
        valid = ("phi_four", "spin")
        if theory not in valid:
            raise ConfigError(
                f"Selected theory: {theory}, has not been implemented yet",
                theory,
                valid,
            )
        return theory

    def parse_theory_N(self, N: int):
        """N for O(N) and CP^{N-1} models"""
        return N

    def produce_target_dimension(self, theory, theory_N = 2):  # requires default N
        """Dimension of the target manifold for the theory.
        This allows for more generality in produce functions than
        if we were to use theory-specific parameters such as
        'theory_N', for example.
        """
        if theory == "phi_four":
            return 1
        elif theory == "spin":
            return theory_N - 1
        elif theory == "cpn":
            return 2 * theory_N - 2  # not yet implemented


    def produce_target_manifold(self, theory):
        """Return a string specifying the class of manifolds (of arbitrary
        dimensionality) to the target manifold for the theory belongs."""
        if theory == "phi_four":
            return "R"
        elif theory in ("spin", "cpn"):
            return "S"

    def produce_config_size(self, lattice_size, target_dimension):
        return target_dimension * lattice_size

    def produce_geometry(self, lattice_length):
        return Geometry2D(lattice_length)

    def parse_m_sq(self, m: (float, int)):
        return m

    def parse_lam(self, lam: (float, int)):
        return lam

    def parse_use_arxiv_version(self, do_use: bool):
        return do_use

    def parse_beta(self, beta: float):
        return beta

    def parse_flow_model(self, mod: str):
        valid = ("real_nvp",)
        if mod not in valid:
            raise ConfigError(
                f"Selected flow model: {mod}, has not been implemented yet", mod, valid,
            )
        return mod

    def parse_base_dist(self, dist):
        valid = ("normal", "uniform")
        if dist not in valid:
            raise ConfigError(
                f"Selected base distribution: {dist}, has not been implemented yet",
                dist,
                valid,
            )
        return dist

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
    def produce_target(self, theory, target_dimension):
        """Return the function which initialises the correct action"""
        if theory == "phi_four":
            return phi_four_action
        elif theory == "spin":
            if target_dimension == 1:
                return xy_hamiltonian
            elif target_dimension == 2:
                return heisenberg_hamiltonian
        raise ConfigError(
            f"Target distribution for theory: {theory}, has not been implemented yet for target dimeinsion {target_dimension}",
        )

    @explicit_node
    def produce_base(self, target_manifold, target_dimension, base_dist="normal"):
        """Return the action which loads appropriate base distribution"""
        if target_manifold == "R":
            if base_dist == "normal":
                return normal_distribution
            elif base_dist == "uniform":
                return uniform_distribution
        elif target_manifold == "S":
            if base_dist == "uniform":
                if target_dimension == 1:
                    return circular_uniform_distribution
                if target_dimension == 2:
                    return spherical_uniform_distribution
        raise ConfigError(
            f"Base distribution: {base_dist}, has not been implemented yet for target manifold {target_manifold}{target_dimension}",
        )

    @explicit_node
    def produce_model(self, target_manifold, target_dimension, flow_model="real_nvp"):
        if flow_model == "real_nvp":
            if target_manifold == "R":
                return real_nvp
            elif target_manifold == "S":
                if target_dimension == 1:
                    return project_circle
                elif target_dimension == 2:
                    return project_sphere
        raise ConfigError(
            f"Flow model: {flow_model}, has not been implemented yet for target manifold {target_manifold}{target_dimension}",
        )

    @explicit_node
    def produce_field_ensemble(self, theory):
        if theory == "phi_four":
            return scalar_field
        elif theory == "xy":
            return xy_field
        elif theory == "heisenberg":
            return heisenberg_field
        raise ConfigError(
            f"Selected theory: {theory}, has not been implemented yet",
            theory,
            ["phi_four", "xy", "heisenberg"],
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
