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

    Intended usage: instantiate class before training phase, passing the
    batch size as a parameter.
    The __call__ method can then be re-used during the sampling phase, but
    with a flexible n_sample parameter to replace n_batch.

    Inputs:
    -------
    field_dimension: int
        Number of independent field components at each lattice site.
        Default = 1, i.e. a scalar field.
    lattice_volume: int
        Number of lattice sites.
    n_batch: int
        Batch size for the training phase, where each member of the batch
        is a field configuration with field_dimension * lattice_volume
        components.
    """

    def __init__(self, n_batch, lattice_volume, field_dimension=1):
        self.n_batch = n_batch
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
        print(self.size_out)
        return log(sqrt(pow(2 * pi, self.size_out)))

    def log_density(self, sample: torch.Tensor) -> torch.Tensor:
        """Return log probability density of a sample generated from
        the __call__ method above with default arguments!

        The size of the sample (number of configurations) should be exactly
        self.n_batch, otherwise the pre-calculated log normalisation will
        be incorrect.

        Return shape: (n_batch, 1).
        """
        exponent = -torch.sum(0.5 * sample.pow(2), dim=1, keepdim=True)
        return exponent - self._log_normalisation
