import numpy as np
from math import pi

class FreeScalarEigenmodes:
    r"""
    The action for the theory of a free scalar on a lattice is
    
        S(\phi) = \frac{1}{2} \sum_x \sum_y \phi(x) K(x, y) \phi(y)
    
    The eigenmodes of the matrix

        K(x, y) = \box(x, y) + m^2 \delta(x - y)
    
    (which is referred to here as the kinetic operator) are the momentum
    states \tilde\phi(p), and the associated eigenvalues in two dimensions
    are

        \lambda(p) = m^2 + 4 \sin^2(p1 / 2) + 4 \sin^2(p2 / 2)

    where (p1, p2) are the two components of p.
    
    It can be shown that the action can be written in Fourier space as
        
        S(\tilde\phi) = \frac{1}{2V} \lambda(p) |\tilde\phi(p)|^2

    and hence the partition function is a product of Gaussian distributions
    for the variables |\tilde\phi(p)|, with variances

        \sigma^2(p) = V / \lambda(p)

    This means we can sample from this probability distribution in Fourier
    space by simply generating Gaussian random numbers.
    """
    def __init__(self, m_sq: int, lattice_length: int):
        self.m_sq = m_sq
        self.lattice_length = lattice_length

        self.lattice_volume = self.lattice_length ** 2
        self.ip0 = self.lattice_length // 2 - 1  # index for zero momentum

        self.momenta = (
            2
            * pi
            / self.lattice_length
            * np.arange(-self.lattice_length // 2 + 1, self.lattice_length // 2 + 1)
        )

        self.mode_is_real = self._mode_is_real()

        self.eigenvalues = self._eigenvalues()
        self.variance = self._variance()

    def _mode_is_real(self):
        """Returns a boolean array where the True components correspond to
        momentum vectors whose eigenmode is purely real.

        These momenta are:
            (0, 0)
            (0, p_max)
            (p_max, 0)
            (p_max, p_max)
        
        where p_max = 2 \pi / L * L / 2 = \pi is the Nyquist frequency for the
        one-dimensional lattice with unit lattice spacing and length L.
        """
        is_real = np.zeros((self.lattice_length, self.lattice_length), dtype=bool)
        is_real[self.ip0, self.ip0] = True  # (0, 0) component
        is_real[self.ip0, -1] = is_real[-1, self.ip0] = True  # (0, p_max) nd (p_max, 0)
        is_real[-1, -1] = True  # (p_max, p_max)
        return is_real

    def _eigenvalues(self):
        """Returns a two-dimensional array whose values are the eigenvalues
        of the lattice kinetic operator for the free scalar theory.
        """
        p1, p2 = np.meshgrid(self.momenta, self.momenta)
        eigvals = 4 - 2 * (np.cos(p1) + np.cos(p2)) + self.m_sq
        return eigvals

    def _variance(self):
        r"""Returns a two-dimensional array whose values are the variances of
        the one-dimensional Gaussian distributions for the *real* components of
        the eigenmodes.

        With the exception of the four purely-real modes, the variance of the
        real component for a given momentum p is equal to the variance for the
        imaginary component.

        First, the variance of the modulus of the eigenmodes is calculated
        by identifying them with the inverse of the eigenvalues of the kinetic
        operator.
        
        Because the real-space fields are real, we have a constraint on the
        eigenmodes:

            \tilde\phi(-p) = \tilde\phi(p)*

        This means that we can write the partition function as a product,
        over positive momenta only, of one-dimensional Gaussian distributions
        for a(p) and b(p), the real and imaginary parts of \tilde\phi(p).
        
        For the purely real eigenmodes, the variance of a(p) is equal to the
        variance of |\tilde\phi(p)|. However, for the complex eigenmodes the
        variance of a(p) and b(p) is half of the variance of |\tilde\phi(p)|.
        """
        variance_mod = self.lattice_volume * np.reciprocal(self.eigenvalues)

        variance = variance_mod
        variance[~self.mode_is_real] /= 2

        return variance

    @staticmethod
    def gen_complex_normal(n_sample, sigma, real=False):
        """Returns a stack of complex arrays where real and imaginary components
        are drawn from a Gaussian distribution with the same width.

        Inputs:
        -------
        n_sample: int
            sample size
        sigma: numpy.ndarray
            array of standard deviations. Need not be one-dimensional
        real: bool
            (optional) flag. If True, the imaginary component is set to
            zero, but a complex array is still returned.

        Returns:
        --------
        out: numpy.ndarray
            complex array of shape (n_sample, *sigma.shape)
        """
        shape_out = np.zeros((n_sample, *(sigma.shape)))
        if real:
            return np.random.normal(loc=shape_out, scale=sigma) + 0j
        else:
            return np.random.normal(loc=shape_out, scale=sigma) + 1j * np.random.normal(
                loc=shape_out, scale=sigma
            )

    def gen_eigenmodes(self, n_sample):
        """Returns sample of eigenmodes for the lattice free scalar theory.

        The real and imaginary components of the eigenmodes are drawn from
        Gaussian distributions with variances given by the eigenvalues of the
        kinetic operator - see _variance() method above.
        
        Inputs:
        -------
        n_sample: int
            sample size

        Returns:
        --------
        eigenmodes: numpy.ndarray
            complex array of eigenmodes with shape (n_sample, L, L)
            where L is the side length of the square lattice.
        """
        eigenmodes = np.empty(
            (n_sample, self.lattice_length, self.lattice_length), dtype=complex
        )
        sigma = self.variance.sqrt()  # standard deviations
        Z = self.ip0  # relabel zero momentum index for convenience

        # Bottom right square (p1, p2 > 0)
        eigenmodes[:, Z:, Z:] = self.gen_complex_normal(n_sample, sigma[Z:, Z:])

        # Four of these components are real
        eigenmodes[:, self.mode_is_real] = self.gen_complex_normal(
            n_sample, sigma[self.mode_is_real], real=True
        )

        # Top right square (p1 < 0, p2 > 0)
        eigenmodes[:, :Z, Z + 1 : -1] = self.gen_complex_normal(
            n_sample, sigma[:Z, Z + 1 : -1]
        )

        # Reflect bottom right to top left
        eigenmodes[:, : Z + 1, : Z + 1] = np.flip(
            eigenmodes[:, Z:-1, Z:-1].conj(), axis=(-2, -1)
        )

        # Reflect top right to bottom left
        eigenmodes[:, Z + 1 : -1, :Z] = np.flip(
            eigenmodes[:, :Z, Z + 1 : -1].conj(), axis=(-2, -1)
        )

        # Reflect row / col with p1 = p_max / p2 = p_max
        eigenmodes[:, :Z, -1] = np.flip(eigenmodes[:, Z + 1 : -1, -1].conj())
        eigenmodes[:, -1, :Z] = np.flip(eigenmodes[:, -1, Z + 1 : -1].conj())

        return eigenmodes

    def gen_real_space_fields(self, n_sample):
        """Returns the inverse fourier transform of a sample of eigenmodes.

        Inputs:
        -------
        n_sample: int
            sample size

        Returns:
        --------
        fields: numpy.ndarray
            real array of real-space fields, with shape (n_sample, L, L),
            where L is the side-length of the square lattice.
        """
        eigenmodes = self.gen_eigenmodes(n_sample)

        # Numpy fft requires input in form
        # [f_0, f_1, ..., f_{n-1}, f_n, f_{-n+1}, ..., f_{-1}]
        eigenmodes = np.roll(eigenmodes, (-self.ip0, -self.ip0), (-2,-1))

        fields = np.fft.ifft2(eigenmodes)
        #TODO add a check that fields are indeed real

        return fields.real

