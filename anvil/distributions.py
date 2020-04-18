"""
distributions.py

Module containing classes corresponding to different base distributions.
"""
from math import pi, log, sqrt
import torch
import torch.nn as nn


class NormalDist:
    """
    Class which handles the generation of a sample of field configurations
    following the standard normal distribution.

    Intended usage: instantiate class before training phase.
    The __call__ method can then be used during sampling since this
    object will be associated with the loaded model.

    Inputs:
    -------
    lattice_size: int
        Number of lattice sites.
    field_dimension: int
        Number of independent field components at each lattice site.
        Default = 1, i.e. a scalar field.
    """

    def __init__(self, lattice_size, field_dimension=1):
        self.lattice_size = lattice_size
        self.field_dimension = field_dimension

        # Number of components per field configuration =
        # size of output Tensor at dimension 1
        self.size_out = self.lattice_size * self.field_dimension

        # Pre-calculate normalisation for log density
        self.log_normalisation = self._log_normalisation()

    def __call__(self, sample_size) -> tuple:
        """Return a tuple (sample, log_density) for a sample of sample_size states
        drawn from the standard uniform distribution.
        
        See docstrings for instance methods generator, log_density for more details.
        """
        sample = self.generator(sample_size)
        log_density = self.log_density(sample)
        return sample, log_density

    def generator(self, sample_size):
        """Return tensor of values drawn from the standard normal distribution
        with mean 0, variance 1.
        
        Return shape: (sample_size, field_dimension * lattice_size).
        """
        return torch.randn(sample_size, self.size_out)

    def _log_normalisation(self) -> float:
        """logarithm of the normalisation for the density function."""
        return log(sqrt(pow(2 * pi, self.size_out)))

    def log_density(self, sample: torch.Tensor) -> torch.Tensor:
        """Return log probability density of a sample generated from
        the __call__ method above.

        Return shape: (sample_size, 1) where sample_size is the number of
        field configurations (the first dimension of sample).
        """
        exponent = -torch.sum(0.5 * sample.pow(2), dim=1, keepdim=True)
        return exponent - self.log_normalisation


def normal_distribution(lattice_size, field_dimension=1):
    """returns an instance of the NormalDist class"""
    return NormalDist(lattice_volume=lattice_size, field_dimension=field_dimension,)


class SphericalUniformDist:
    """
    Class which handles the generation of a sample of field configurations
    following the uniform distribution on a unit sphere.

    Inputs:
    -------
    lattice_size: int
        Number of lattice sites.
    field_dimension: int
        Number of angles parameterising points on the unit sphere at each
        lattice site. Equal to N-1 where N is the number of components
        in the Euclidean vector describing a given point.
        Default = 1, i.e. a two-component vector field defined on the
        unit circle.
    """

    def __init__(self, lattice_size, field_dimension=1):
        self.lattice_size = lattice_size
        self.field_dimension = field_dimension

        # Number of components per field configuration =
        # size of output Tensor at dimension 1
        self.size_out = self.lattice_size * self.field_dimension

        if self.field_dimension is 1:
            self.generator = self.gen_circular
            # Overrides the existing method to save time
            self.log_volume_element = lambda sample: torch.zeros((sample.shape[0], 1))
        elif self.field_dimension is 2:
            self.generator = self.gen_spherical
        else:
            self.generator = self.gen_hyperspherical

        # Powers of sin(angle) for each angle except the azimuth
        # in the probability density function
        self._sin_pow = torch.arange(self.field_dimension - 1, 0, -1).view(1, 1, -1)

    def __call__(self, sample_size):
        """Return tensor of values drawn from uniform distribution
        on a unit sphere of dimension field_dimension.
        
        Return shape: (sample_size, field_dimension * lattice_size).
        """
        sample = self.generator(sample_size)
        log_density = self.log_density(sample)
        return sample, log_density

    def gen_circular(self, sample_size) -> torch.Tensor:
        """Return tensor of values distributed uniformly on the unit 1-sphere.
        """
        return torch.rand(sample_size, self.lattice_size) * 2 * pi

    def gen_spherical(self, sample_size) -> torch.Tensor:
        r"""Return tensor of values distributed uniformly on the unit 2-sphere.
        
        Uses inversion sampling to map random variables x ~ [0, 1] to the 
        polar angle \theta which has the marginalised density \sin\theta,
        via the inverse of its cumulative distribution.

                        \theta = \arccos( 1 - 2 x )
        """
        sample = torch.stack(
            (
                torch.acos(1 - 2 * torch.rand(sample_size, self.lattice_size)),
                torch.rand(sample_size, self.lattice_size) * 2 * pi,
            ),
            dim=-1,
        )
        return sample.view(sample_size, 2 * self.lattice_size)

    def gen_hyperspherical(self, sample_size) -> torch.Tensor:
        """Return values distributed uniformly on the unit N-sphere.
        
        Uses Marsaglia's algorithm.
        """
        rand_normal = torch.randn(sample_size, self.lattice_size, self.dimension + 1)
        points = rand_normal / rand_normal.norm(dim=1, keepdim=True)

        # TODO: convert to self.dimension angles. Kind of a faff and not urgent
        raise NotImplementedError

    def log_volume_element(self, sample: torch.Tensor) -> torch.Tensor:
        r"""Return log of volume element for the probability measure, such that
        total probability mass equals 1.

        If the volume element is \prod_{n=1}^V \Omega_n and the density function is
        is e^{iS(\phi)}, then the probability measure (differential probability mass)
        
            \prod_{n=1}^V \Omega_n e^{iS(\phi)} d^{N-1}\phi_n^i

        must give 1 when integrated over the entire space of configurations.
        This means we should take
                
                p(\phi) = \prod_{n=1}^V \Omega_n e^{iS(\phi)}

        as defining our target probability density.

        In this case, \Omega_n arises from the use of spherical polar coordinates to
        parameterise the fields, and is equal to the surface area element for the
        unit (N-1)-sphere expressed in these coordinates, where (N-1) is equal to
        self.field_dimension.
        
        This also goes by the name of the Jacobian determinant for the change
        of coordinates Euclidean -> spherical.

        Inputs
        ------
        sample: torch.Tensor
            A tensor of shape (sample_size, lattice_size * field_dimension).
            When reshaped into (sample_size, lattice_size, field_dimension),
            the final dimension should contain field_dimension angles from the
            same lattice site, with the largest index corresponding to the
            azimuthal angle.

        Returns
        -------
        torch.Tensor which is the logarithm of the Jacobian determinant for
        the entire lattice, with shape (sample_size, 1).
        """
        log_jacob = (
            self._sin_pow
            * torch.log(
                torch.sin(
                    sample.view(-1, self.lattice_size, self.field_dimension)[:, :, :-1]
                )
            )
        ).sum(
            dim=2
        )  # sum over all angles except azimuth
        return log_jacob.sum(dim=1, keepdim=True)  # sum over volume

    def log_density(self, sample: torch.Tensor) -> torch.Tensor:
        """Return log probability density of a sample generated from
        the __call__ method above.

        Note that this is equivalent to the surface area element for
        the (N-1)-sphere expressed in spherical coordinates.
        """
        return self.log_volume_element(sample)


def spherical_distribution(lattice_size, field_dimension=1):
    return SphericalUniformDist(lattice_size, field_dimension)


