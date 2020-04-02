"""
distributions.py

Module containing classes corresponding to different base distributions.
"""
from math import pi, log, sqrt
import torch


class NormalDist:
    """
    Class which handles the generation of a sample of field configurations
    following the standard normal distribution.

    Intended usage: instantiate class before training phase.
    The __call__ method can then be used during sampling since this
    object will be associated with the loaded model.

    Inputs:
    -------
    field_dimension: int
        Number of independent field components at each lattice site.
        Default = 1, i.e. a scalar field.
    lattice_volume: int
        Number of lattice sites.
    """

    def __init__(self, lattice_volume, field_dimension=1):
        self.lattice_volume = lattice_volume
        self.field_dimension = field_dimension

        # Number of components per field configuration =
        # size of output Tensor at dimension 1
        self.size_out = self.lattice_volume * self.field_dimension

        # Pre-calculate normalisation for log density
        self._log_normalisation = self.log_normalisation()

    def __call__(self, n_sample) -> torch.Tensor:
        """Return tensor of values drawn from the standard normal distribution
        with mean 0, variance 1.
        
        Return shape: (n_sample, field_dimension * lattice_volume).
        """
        return torch.randn(n_sample, self.size_out)

    def log_normalisation(self) -> float:
        """logarithm of the normalisation for the density function."""
        return log(sqrt(pow(2 * pi, self.size_out)))

    def log_density(self, sample: torch.Tensor) -> torch.Tensor:
        """Return log probability density of a sample generated from
        the __call__ method above.

        Return shape: (n_sample, 1) where n_sample is the number of
        field configurations (the first dimension of sample).
        """
        exponent = -torch.sum(0.5 * sample.pow(2), dim=1, keepdim=True)
        return exponent - self._log_normalisation
