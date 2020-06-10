"""
distributions.py

Module containing classes corresponding to different probability distributions.
"""
from math import pi, log, sqrt
import torch
import torch.nn as nn

from scipy.special import i0


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
        Number of nodes on the lattice.
    sigma: float
        Standard deviation for the distribution.
    mean: float
        Mean for the distribution.
    """

    def __init__(self, lattice_size, *, sigma, mean):
        self.size_out = lattice_size
        self.sigma = sigma
        self.mean = mean

        self.exp_coeff = 1 / (2 * self.sigma ** 2)

        # Pre-calculate normalisation for log density
        self.log_normalisation = self.size_out * log(sqrt(2 * pi) * self.sigma)

    def __call__(self, sample_size) -> tuple:
        """Return a tuple (sample, log_density) for a sample of 'sample_size'
        states drawn from the normal distribution.
        
        Return shape: (sample_size, lattice_size) for the sample,
        (sample_size, 1) for the log density.
        """
        sample = torch.empty(sample_size, self.size_out).normal_(
            mean=self.mean, std=self.sigma
        )

        return sample, self.log_density(sample)

    def log_density(self, sample):
        """Logarithm of the pdf, calculated for a given sample. Dimensions (sample_size, 1)."""
        exponent = -self.exp_coeff * torch.sum(
            (sample - self.mean).pow(2), dim=1, keepdim=True
        )
        return exponent - self.log_normalisation

    @property
    def pdf(self):
        x = torch.linspace(-5 * self.sigma, 5 * self.sigma, 10000)
        return (
            (
                x,
                torch.exp(-self.exp_coeff * (x - self.mean) ** 2)
                / (sqrt(2 * pi) * self.sigma),
            ),
        )


class UniformDist:
    """Class which handles the generation of a sample of field configurations
    following the uniform distribution on some interval.

    Inputs:
    -------
    lattice_size: int
        Number of nodes on the lattice.
    support: tuple
        Low and high limits for the interval.
    """

    def __init__(self, lattice_size, *, support):
        self.size_out = lattice_size

        self.x_min, self.x_max = support

        self._log_density = -log((self.x_max - self.x_min) * lattice_size)  # normalised
        self.log_density = (
            lambda sample: torch.ones((sample.shape[0], 1)) * self._log_density
        )

    def __call__(self, sample_size):
        """Return a tuple (sample, log_density) for a sample of 'sample_size'
        states drawn from a uniform distribution.
        
        Return shape: (sample_size, lattice_size) for the sample,
        (sample_size, 1) for the log density.
        """
        sample = torch.empty(sample_size, self.size_out).uniform_(
            self.x_min, self.x_max
        )
        return sample, self.log_density(sample)

    @property
    def pdf(self):
        x = torch.linspace(self.x_min, self.x_max, 10000)
        return ((x, torch.zeros_like(x) + 1 / (self.x_max - self.x_min)),)


class SemicircleDist:
    """Class which handles the generation of a sample of field configurations
    following the Wigner semicircle distribution.

    Inputs:
    -------
    lattice_size: int
        Number of nodes on the lattice.
    radius: (int, float)
        radius of semicircle
    mean: (int, float)
        location of center of distribution. Not really useful.
    """

    def __init__(self, lattice_size, *, radius, mean):
        self.size_out = lattice_size
        self.radius = radius
        self.mean = mean

        self.log_normalisation = self.size_out * log((pi * self.radius ** 2) / 2)

    def __call__(self, sample_size):
        """Return a tuple (sample, log_density) for a sample of 'sample_size'
        states drawn from the semicircle distribution.
        
        Return shape: (sample_size, lattice_size) for the sample,
        (sample_size, 1) for the log density.
        """
        sample = (
            self.radius
            * torch.sqrt(torch.empty(sample_size, self.size_out).uniform_())
            * torch.cos(torch.empty(sample_size, self.size_out).uniform_(0, pi))
            + self.mean
        )
        return sample, self.log_density(sample)

    def log_density(self, sample):
        """Logarithm of the pdf, calculated for a given sample. Dimensions (sample_size, 1)."""
        return (
            torch.sum(
                0.5 * torch.log(self.radius ** 2 - (sample - self.mean) ** 2),
                dim=1,
                keepdim=True,
            )
            - self.log_normalisation
        )

    @property
    def pdf(self):
        x = torch.linspace(-self.radius, self.radius, 10000)
        dens = 2 / (pi * self.radius ** 2) * torch.sqrt(self.radius ** 2 - x ** 2)
        return ((x + self.mean, dens),)


class VonMisesDist:
    """Class implementing the von Mises distribution, which is the
    circular analogue of the normal distribution.

    The von Mises distribution has two parameters: a 'contentration'
    and a 'location'. The location is the mean '\mu', directly analogous to
    the normal case. The concentration '\kappa' parameterises the
    sharpness of the peak, and is analogous to the inverse of the
    variance of the normal distribution.

    The probability density function is:

        p(x) = \exp( \kappa * \cos(x - \mu) ) / ( 2 * pi * I_0(\kappa) )

    where I_0(\kappa) is the order-0 modified Bessel function of the
    first kind.

    Inputs:
    -------
    lattice_size: int
        number of nodes on the lattice.
    concentration: float
        parameter dictating sharpness of the peak.
    mean: float
        mean of the distribution.

    Notes:
    ------
    The von Mises distribution was implemented in PyTorch 1.5 as a
    torch.distribution object. This class currently uses the PyTorch
    implementation to draw a random sample, but does not use it for
    the log density calculation. There's no good reason for this other
    than it's nice to see the calculation written out.
    """

    def __init__(self, lattice_size, *, concentration, mean):
        self.size_out = lattice_size
        self.kappa = concentration
        self.mean = mean

        self.log_normalisation = self.size_out * log(2 * pi * i0(self.kappa))

        self.generator = torch.distributions.von_mises.VonMises(
            loc=self.mean, concentration=self.kappa
        ).sample

    def __call__(self, sample_size):
        """Return a tuple (sample, log_density) for a sample of 'sample_size'
        states drawn from the von Mises distribution.
        
        Return shape: (sample_size, lattice_size) for the sample,
        (sample_size, 1) for the log density.
        """
        sample = self.generator((sample_size, self.size_out)) + pi  # [0, 2\pi)
        log_density = self.log_density(sample)
        return sample, log_density

    def log_density(self, sample):
        """Logarithm of the pdf, calculated for a given sample. Dimensions (sample_size, 1)."""
        return (
            self.kappa * torch.cos(sample - self.mean).sum(dim=1, keepdim=True)
            - self.log_normalisation
        )

    @property
    def pdf(self):
        x = torch.linspace(0, 2 * pi, 10000)
        return (
            (
                x,
                torch.exp(self.kappa * torch.cos(x - self.mean))
                / (2 * pi * i0(self.kappa)),
            ),
        )


class SphericalUniformDist:
    """
    Class which handles the generation of a sample of field configurations
    following the uniform distribution on a unit sphere.

    Inputs:
    -------
    lattice_size: int
        number of nodes on the lattice
    """

    def __init__(self, lattice_size):
        # Two components for each lattice site
        self.lattice_size = lattice_size
        self.size_out = lattice_size * 2

    def __call__(self, sample_size):
        r"""Return tensor of values drawn from uniform distribution
        on a unit 2-dimensional sphere, along with the corresponding
        log probability density.
        
        Return shape: (sample_size, lattice_size) for the sample,
        (sample_size, 1) for the log density.
        
        Notes
        -----
        Uses inversion sampling to map random variables x ~ [0, 1] to the 
        polar angle \theta which has the marginalised density \sin\theta,
        via the inverse of its cumulative distribution.

                        \theta = \arccos( 1 - 2 x )
        """
        polar = torch.acos(1 - 2 * torch.rand(sample_size, self.lattice_size))
        azimuth = torch.rand(sample_size, self.lattice_size) * 2 * pi

        # Quicker to do this than call log_density method
        log_density = torch.log(torch.sin(polar)).sum(dim=1, keepdim=True)

        sample = torch.stack((polar, azimuth), dim=-1).view(-1, self.size_out)

        return sample, log_density

    def log_density(self, sample):
        r"""Takes a sample of shape (sample_size, lattice_size) and
        computes the logarithm of the probability density function for
        the spherical uniform distribution.

        It is assumed that the tensor follows the __call__ method above
        in that, when a view of shape (sample_size, lattice_size / 2, 2)
        is taken, the 0th element in the 2nd dimension is the polar angle.
        In other words, every second element of the input tensor, starting
        at the 0th element, is a polar angle.
        
        The density function is equal to the surface area element
        for the 2-sphere expressed in spherical coordinates, which,
        for lattice site 'n' containing polar angle '\theta_n', is

                    | \det J_n | = \sin \theta_n 
        """
        return torch.log(torch.sin(sample[:, ::2])).sum(dim=1, keepdim=True)

    @property
    def pdf(self):
        pol = torch.linspace(0, pi, 10000)
        az = torch.linspace(0, 2 * pi, 10000)
        return (pol, torch.sin(pol)), (az, torch.zeros_like(az) + 1 / (2 * pi))


def standard_normal_distribution(lattice_size):
    """returns an instance of the NormalDist class with mean 0 and
    variance 1"""
    return NormalDist(lattice_size, sigma=1, mean=0)


def normal_distribution(lattice_size, sigma=1, mean=0):
    """Returns an instance of the NormalDist class"""
    return NormalDist(lattice_size, sigma=sigma, mean=mean)


def uniform_distribution(lattice_size, support=(-1, 1)):
    """Returns an instance of the UniformDist class.

    The default interval is intentionally zero-centered, anticipating use
    as a base distribution."""
    return UniformDist(lattice_size, support=support)


def standard_uniform_distribution(lattice_size):
    """Returns an instance of the UniformDist class with interval [0, 1)"""
    return UniformDist(lattice_size, support=(0, 1))


def circular_uniform_distribution(lattice_size):
    """Returns an instance of the UniformDist class with interval [0, 2 * pi)"""
    return UniformDist(lattice_size, support=(0, 2 * pi))


def von_mises_distribution(lattice_size, concentration=1, mean=0):
    """Returns and instance of the VonMisesDist class."""
    return VonMisesDist(lattice_size, concentration=concentration, mean=mean)


def spherical_uniform_distribution(lattice_size):
    """Returns an instance of the SphericalUniformDist class"""
    return SphericalUniformDist(lattice_size)


def semicircle_distribution(lattice_size, radius=pi, mean=0):
    """Returns an instance of the SemicircleDist class."""
    return SemicircleDist(lattice_size, radius=radius, mean=mean)


class PhiFourAction:
    """Return the phi^4 action given either a single state size
    (1, length * length) or a stack of N states (N, length * length).
    See Notes about action definition.

    The forward pass returns the corresponding log density (unnormalised) which
    is equal to -S

    Parameters
    ----------
    geometry:
        define the geometry of the lattice, including dimension, size and
        how the state is split into two parts
    m_sq: float
        the value of the bare mass squared
    lam: float
        the value of the bare coupling

    Examples
    --------
    Consider the toy example of this class acting on a random state

    >>> geom = Geometry2D(2)
    >>> action = PhiFourAction(1, 1, geom)
    >>> state = torch.rand((1, 2*2))
    >>> action(state)
    tensor([[-2.3838]])
    >>> state = torch.rand((5, 2*2))
    >>> action(state)
    tensor([[-3.9087],
            [-2.2697],
            [-2.3940],
            [-2.3499],
            [-1.9730]])

    Notes
    -----
    that this is the action as defined in
    https://doi.org/10.1103/PhysRevD.100.034515 which might differ from the
    current version on the arxiv.

    """

    def __init__(self, m_sq, lam, geometry, use_arxiv_version=False):
        super(PhiFourAction, self).__init__()
        self.shift = geometry.get_shift()
        self.lam = lam
        self.m_sq = m_sq
        self.lattice_size = geometry.length ** 2
        if use_arxiv_version:
            self.version_factor = 2
        else:
            self.version_factor = 1

    def log_density(self, phi_state: torch.Tensor) -> torch.Tensor:
        """Perform forward pass, returning -action for stack of states. Note
        here the minus sign since we want to return the log density of the
        corresponding unnormalised distribution

        see class Notes for details on definition of action.
        """
        action = (
            self.version_factor * (2 + 0.5 * self.m_sq) * phi_state ** 2  # phi^2 terms
            + self.lam * phi_state ** 4  # phi^4 term
            - self.version_factor
            * torch.sum(
                phi_state[:, self.shift] * phi_state.view(-1, 1, self.lattice_size),
                dim=1,
            )  # derivative
        ).sum(
            dim=1, keepdim=True  # sum across sites
        )
        return -action


class O2Action:
    r"""
    The (shifted) action for the O(2) non-linear sigma model, calculated
    from a stack of polar angles with shape (sample_size, lattice_size).
    
    The action is shifted by -2 * V * \beta, making it equivalent to \beta
    times the Hamiltonian for the classical XY spin model.

    The fields or 'spins' are defined as having modulus 1, such that they
    take values on the unit circle.

    Parameters
    ----------
    geometry:
        define the geometry of the lattice, including dimension, size and
        how the state is split into two parts
    beta: float
        the inverse temperature (coupling strength).
    """

    def __init__(self, beta, geometry):
        super().__init__()
        self.beta = beta
        self.lattice_size = geometry.length ** 2
        self.shift = geometry.get_shift()

    def log_density(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute action from a stack of angles (not Euclidean field components)
        with shape (sample_size, lattice_size).
        """
        action = -self.beta * torch.cos(
            state[:, self.shift] - state.view(-1, 1, self.lattice_size)
        ).sum(
            dim=1,
        ).sum(  # sum over two shift directions (+ve nearest neighbours)
            dim=1, keepdim=True
        )  # sum over lattice sites
        return -action


class O3Action:
    r"""
    The (shifted) action for the O(3) non-linear sigma model, calculated from
    a stack of polar and azimuthal angles with shape
    (sample_size, 2 * lattice_size).

    The action is shifted by -2 * V * \beta, making it equivalent to \beta
    times the Hamiltonian for the classical Heisenberg spin model.

    The field or 'spins' are defined as having modulus 1, such that they take
    values on the unit 2-sphere, and can be parameterised by two angles using
    spherical polar coordinates (with the radial coordinate equal to one).

    Parameters
    ----------
    geometry:
        define the geometry of the lattice, including dimension, size and
        how the state is split into two parts
    beta: float
        the inverse temperature (coupling strength).
    """

    def __init__(self, beta, geometry):
        super().__init__()
        self.beta = beta
        self.lattice_size = geometry.length ** 2
        self.shift = geometry.get_shift()

    def log_density(self, state: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the O(3) action from a stack of angles with shape
        (sample_size, 2 * volume).

        Also computes the logarithm of the 'volume element' for the probability
        distribution due to parameterisating the spin vectors using polar coordinates.
        
        The volume element for a configuration is a product over all lattice sites
        
            \prod_{n=1}^V sin(\theta_n)

        where \theta_n is the polar angle for the spin at site n.
        
        Notes
        -----
        Assumes that state.view(-1, lattice_size, 2) yields a tensor for which the
        two elements in the final dimension represent, respectively, the polar and
        azimuthal angles for the same lattice site.
        """
        polar = state[:, ::2]
        azimuth = state[:, 1::2]
        cos_polar = torch.cos(polar)
        sin_polar = torch.sin(polar)

        action = -self.beta * (
            cos_polar[:, self.shift] * cos_polar.view(-1, 1, self.lattice_size)
            + sin_polar[:, self.shift]
            * sin_polar.view(-1, 1, self.lattice_size)
            * torch.cos(azimuth[:, self.shift] - azimuth.view(-1, 1, self.lattice_size))
        ).sum(
            dim=1,
        ).sum(  # sum over two shift directions (+ve nearest neighbours)
            dim=1, keepdim=True
        )  # sum over lattice sites

        log_volume_element = torch.log(sin_polar).sum(dim=1, keepdim=True)

        return log_volume_element - action


def phi_four_action(m_sq, lam, geometry, use_arxiv_version):
    """returns instance of PhiFourAction"""
    return PhiFourAction(
        m_sq, lam, geometry=geometry, use_arxiv_version=use_arxiv_version
    )


def o2_action(beta, geometry):
    return O2Action(beta, geometry)


def o3_action(beta, geometry):
    return O3Action(beta, geometry)


BASE_OPTIONS = {
    "standard_normal": standard_normal_distribution,
    "normal": normal_distribution,
    "uniform": uniform_distribution,
    "standard_uniform": standard_uniform_distribution,
    "circular_uniform": circular_uniform_distribution,
    "von_mises": von_mises_distribution,
    "spherical_uniform": spherical_uniform_distribution,
    "semicircle": semicircle_distribution,
}
TARGET_OPTIONS = dict(
    {"phi_four": phi_four_action, "o2": o2_action, "o3": o3_action,}, **BASE_OPTIONS
)
