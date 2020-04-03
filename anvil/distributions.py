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
    lattice_volume: int
        Number of lattice sites.
    field_dimension: int
        Number of independent field components at each lattice site.
        Default = 1, i.e. a scalar field.
    """

    def __init__(self, lattice_volume, field_dimension=1):
        self.lattice_volume = lattice_volume
        self.field_dimension = field_dimension

        # Number of components per field configuration =
        # size of output Tensor at dimension 1
        self.size_out = self.lattice_volume * self.field_dimension

        # Pre-calculate normalisation for log density
        self.log_normalisation = self._log_normalisation()

    def __call__(self, n_sample) -> torch.Tensor:
        """Return tensor of values drawn from the standard normal distribution
        with mean 0, variance 1.
        
        Return shape: (n_sample, field_dimension * lattice_volume).
        """
        return torch.randn(n_sample, self.size_out)

    def _log_normalisation(self) -> float:
        """logarithm of the normalisation for the density function."""
        return log(sqrt(pow(2 * pi, self.size_out)))

    def log_density(self, sample: torch.Tensor) -> torch.Tensor:
        """Return log probability density of a sample generated from
        the __call__ method above.

        Return shape: (n_sample, 1) where n_sample is the number of
        field configurations (the first dimension of sample).
        """
        exponent = -torch.sum(0.5 * sample.pow(2), dim=1, keepdim=True)
        return exponent - self.log_normalisation


class SphericalUniformDist:
    """
    Class which handles the generation of a sample of field configurations
    following the uniform distribution on a unit sphere.

    Inputs:
    -------
    lattice_volume: int
        Number of lattice sites.
    field_dimension: int
        Number of angles parameterising points on the unit sphere at each
        lattice site. Equal to N-1 where N is the number of components
        in the Euclidean vector describing a given point.
        Default = 1, i.e. a two-component vector field defined on the
        unit circle.
    """

    def __init__(self, lattice_volume, field_dimension=1):
        self.lattice_volume = lattice_volume
        self.field_dimension = field_dimension

        # Number of components per field configuration =
        # size of output Tensor at dimension 1
        self.size_out = self.lattice_volume * self.field_dimension

        if self.field_dimension is 1:
            self.generator = self.gen_circular
        elif self.field_dimension is 2:
            self.generator = self.gen_spherical
        else:
            self.generator = self.gen_hyperspherical

        # Powers of sin(angle) for each angle except the azimuth
        # in the probability density function
        self._sin_pow = torch.arange(self.field_dimension - 1, 0, -1).view(1, -1, 1)

    def __call__(self, n_sample):
        """Return tensor of values drawn from uniform distribution
        on a unit sphere of dimension field_dimension.
        
        Return shape: (n_sample, field_dimension * lattice_volume).
        """
        return self.generator(n_sample)

    def gen_circular(self, n_sample) -> torch.Tensor:
        """Return tensor of values distributed uniformly on the unit 1-sphere.
        """
        return torch.rand(n_sample, self.lattice_volume) * 2 * pi

    def gen_spherical(self, n_sample) -> torch.Tensor:
        r"""Return tensor of values distributed uniformly on the unit 2-sphere.
        
        Uses inversion sampling to map random variables x ~ [0, 1] to the 
        polar angle \theta which has the marginalised density \sin\theta,
        via the inverse of its cumulative distribution.

                        \theta = \arccos( 1 - 2 x )
        """
        sample = torch.cat(
            (
                torch.acos(1 - 2 * torch.rand(n_sample, 1, self.lattice_volume)),
                torch.rand(n_sample, 1, self.lattice_volume) * 2 * pi,
            ),
            dim=1,
        )
        return sample.view(n_sample, 2 * self.lattice_volume)

    def gen_hyperspherical(self, n_sample) -> torch.Tensor:
        """Return values distributed uniformly on the unit N-sphere.
        
        Uses Marsaglia's algorithm.
        """
        rand_normal = torch.randn(n_samples, self.dimension + 1, self.lattice_volume)
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
        
        Return shape: (n_sample, 1) where n_sample is the number of
        field configurations (the first dimension of sample).
        """
        log_jacob = (
            self._sin_pow
            * torch.log(
                torch.sin(
                    sample.view(-1, self.field_dimension, self.lattice_volume)[
                        :, :-1, :
                    ]
                )
            )
        ).sum(
            dim=1
        )  # sum over all angles except azimuth
        return log_jacob.sum(dim=1, keepdim=True)  # sum over volume

    def log_density(self, sample: torch.Tensor) -> torch.Tensor:
        """Return log probability density of a sample generated from
        the __call__ method above.

        Note that this is equivalent to the surface area element for
        the (N-1)-sphere expressed in spherical coordinates.
        """
        return self.volume_element(sample)
