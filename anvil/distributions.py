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
    lattice_size: int
        Number of lattice sites.
    """

    def __init__(self, lattice_size):
        self.size_out = lattice_size

        # Pre-calculate normalisation for log density
        self.log_normalisation = log(sqrt(pow(2 * pi, self.size_out)))

    def __call__(self, sample_size) -> tuple:
        """Return a tuple (sample, log_density) for a sample of "sample_size"
        states drawn from the standard uniform distribution with mean 0,
        variance 1.
        """
        sample = torch.randn(sample_size, self.size_out)
        exponent = -torch.sum(0.5 * sample.pow(2), dim=1, keepdim=True)

        return sample, exponent - self.log_normalisation


class UniformDist:
    """Class which handles the generation of a sample of field configurations
    following the uniform distribution on some interval.

    Inputs:
    -------
    lattice_size: int
        Number of lattice sites.
    interval: tuple
        Low and high limits for the interval
    """

    def __init__(self, lattice_size, interval: tuple = (0, 1)):
        self.size_out = lattice_size

        self.x_min, self.x_max = interval
        self.x_range = self.x_max - self.x_min

    def __call__(self, sample_size):
        """Return tensor of values drawn from uniform distribution.
        
        Return shape: (sample_size, lattice_size).
        """
        sample = torch.rand(sample_size, self.size_out) * self.x_range - self.x_min
        return sample, torch.zeros((sample_size, 1))


class SphericalUniformDist:
    """
    Class which handles the generation of a sample of field configurations
    following the uniform distribution on a unit sphere.

    Inputs:
    -------
    lattice_size: int
        Number of lattice sites.
    """

    def __init__(self, lattice_size):
        self.lattice_size = lattice_size

        # Number of components per field configuration =
        # size of output Tensor at dimension 1
        self.size_out = self.lattice_size * 2

    def __call__(self, sample_size):
        r"""Return tensor of values drawn from uniform distribution
        on a unit 2-dimensional sphere.
        
        Return shape: (sample_size, 2 * lattice_size).
        
        Notes
        -----
        Uses inversion sampling to map random variables x ~ [0, 1] to the 
        polar angle \theta which has the marginalised density \sin\theta,
        via the inverse of its cumulative distribution.

                        \theta = \arccos( 1 - 2 x )
    
        """
        polar = torch.acos(1 - 2 * torch.rand(sample_size, self.lattice_size))
        azimuth = torch.rand(sample_size, self.lattice_size) * 2 * pi

        log_density = torch.log(torch.sin(polar)).sum(dim=1, keepdim=True)

        sample = torch.stack((polar, azimuth), dim=-1).view(-1, self.size_out)

        return sample, log_density


def normal_distribution(lattice_size):
    """returns an instance of the NormalDist class"""
    return NormalDist(lattice_size=lattice_size)


def uniform_distribution(lattice_size, interval=(0, 1)):
    return UniformDist(lattice_size=lattice_size, interval=interval)


def circular_uniform_distribution(lattice_size):
    return UniformDist(lattice_size=lattice_size, interval=(0, 2 * pi))


def spherical_uniform_distribution(lattice_size):
    return SphericalUniformDist(lattice_size)
