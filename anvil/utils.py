"""
utils.py

Module containing assorted useful functions
"""

from math import pi
import torch


class SphericalUniformDist:
    def __init__(self, dimension, lattice_length):
        self.dimension = dimension
        self.lattice_volume = lattice_length ** 2

        if self.dimension is 1:
            self.generator = self.circular_unif
        elif self.dimension is 2:
            self.generator = self.spherical_unif
        else:
            self.generator = self.hyperspherical_unif

    def __call__(self, n_samples):
        return self.generator(n_samples)

    def circular_unif(self, n_samples) -> torch.Tensor:
        """Return samples distributed uniformly on the unit 1-sphere.

        Returns tensor of shape (n_samples, lattice_volume)
        """
        return torch.rand(n_samples, self.lattice_volume) * 2 * pi

    def spherical_unif(self, n_samples) -> torch.Tensor:
        """Return samples distributed uniformly on the unit 2-sphere.

        Uses inversion sampling.

        Returns tensor of shape (n_samples, 2 * lattice_volume), but note that
        this is a reshape of (n_samples, 2, lattice_volume).
        """
        samples = torch.empty(n_samples, 2, self.lattice_volume)
        samples[:, 0, :] = torch.acos(
            1 - 2 * torch.rand(n_samples, self.lattice_volume)
        )
        samples[:, 1, :] = torch.rand(n_samples, self.lattice_volume) * 2 * pi
        return samples.view(n_samples, 2 * self.lattice_volume)

    def hyperspherical_unif(self, n_samples) -> torch.Tensor:
        """Return samples distributed uniformly on the unit N-sphere.
        
        Uses Marsaglia's algorithm.
        """
        rand_normal = torch.randn(n_samples, self.dimension + 1, self.lattice_volume)
        points = rand_normal / rand_normal.norm(dim=1, keepdim=True)

        #TODO: convert to self.dimension angles. Kind of a faff and not urgent
        raise NotImplementedError
        


        
