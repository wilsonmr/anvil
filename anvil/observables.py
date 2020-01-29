"""
observables.py

functions for calculating observables on a stack of states.

Notes
-----
Check the definitions of functions, most are defined according the the arxiv
version: https://arxiv.org/pdf/1904.12072.pdf

"""
import numpy as np
from scipy.signal import correlate
import torch


def arcosh(x):
    """Inverse hyperbolic cosine function for torch.Tensor arguments.

        arcosh(x) = log_e (x + sqrt(x^2 - 1) )
                  = log_e (x) + log_e (1 + sqrt(x^2 - 1) / x)
    """
    c0 = torch.log(x)
    c1 = torch.log1p(torch.sqrt(x * x - 1) / x)
    return c0 + c1


###############################
#####     Observables     #####
###############################
class TwoPointFunction:
    def __init__(self, states, geometry, bootstrap_n_samples):
        self.geometry = geometry
        self.states = states
        self.n_samples = bootstrap_n_samples
        self.sample_size = states.size(0)

        # Bootstrap samples of size (n_samples, sample_size, n_states)
        self.sample_indices = np.random.choice(
            self.sample_size, (self.n_samples, self.sample_size), replace=True
        )

    def __call__(self, x_0: int, x_1: int):
        r"""Calculates the two point connected green function given a set of
        states G(x) where x = (x_0, x_1) refers to a shift applied to the fields
        \phi

        Parameters
        ----------
        x_0: int
            shift of dimension 0
        x_1: int
            shift of dimension 1

        Returns
        -------
        g_func: torch.Tensor
            1d tensor of size (bootstrap_n_samples + 1) containing values of green_function G(x)
            where the 0th element is the mean value, and the others are values
            computed using bootstrap samples.

        """
        shift = self.geometry.get_shift(shifts=((x_0, x_1),), dims=((0, 1),)).view(
            -1
        )  # make 1d

        # Sample of size (target_length, n_states)
        phi = self.states
        phi_shift = phi[:, shift]

        phi_boot = self.states[self.sample_indices, :]
        phi_shift_boot = phi_boot[:, :, shift]

        #  Average over stack of states
        phi_mean = phi.mean(dim=0)
        phi_shift_mean = phi_shift.mean(dim=0)
        phi_shift_phi_mean = (phi_shift * phi).mean(dim=0)
        phi_boot_mean = phi_boot.mean(dim=1)
        phi_shift_boot_mean = phi_shift_boot.mean(dim=1)
        phi_shift_phi_boot_mean = (phi_shift_boot * phi_boot).mean(dim=1)

        # Average over coordinates
        g_func = torch.mean(phi_shift_phi_mean - phi_shift_mean * phi_mean, dim=0)
        g_func_boot = torch.mean(
            phi_shift_phi_boot_mean - phi_shift_boot_mean * phi_boot_mean, dim=1
        )

        return torch.cat((g_func.view(1), g_func_boot))


class VolumeAveraged2pf:
    def __init__(self, states, geometry):
        self.geometry = geometry
        self.states = states

    def __call__(self, x_0: int, x_1: int):
        """
        Return torch Tensor of volume-averaged two point functions, i.e.
        where <\phi(x)> is a mean over points within a single configuration.

        Parameters
        ----------
        x_0: int
            shift of dimension 0
        x_1: int
            shift of dimension 1

        Returns
        -------
        va_2pf: torch.Tensor
            A 1d Tensor containing the volume-averaged two point function
            for each state in the sample
        """
        shift = self.geometry.get_shift(shifts=((x_0, x_1),), dims=((0, 1),)).view(-1)

        va_2pf = (self.states[:, shift] * self.states).mean(dim=1) - self.states.mean(
            dim=1
        ).pow(2)

        return va_2pf


def two_point_function(
    sample_training_output, training_geometry, bootstrap_n_samples=100,
):
    r"""Return instance of TwoPointFunction which can be used to calculate the
    two point green function for a given seperation
    """
    return TwoPointFunction(
        sample_training_output, training_geometry, bootstrap_n_samples,
    )


def volume_averaged_2pf(sample_training_output, training_geometry):
    r"""Return instance ot VolumeAveraged2pf"""
    return VolumeAveraged2pf(sample_training_output, training_geometry)


def zero_momentum_2pf(training_geometry, two_point_function, bootstrap_n_samples=100):
    r"""Calculate the zero momentum green function as a function of t
    \tilde{G}(t, 0) which is assumed to be in the first dimension defined as

        \tilde{G}(t, 0) = 1/L \sum_{x_1} G(t, x_1)

    Returns
    -------
    g_func_zeromom: torch.Tensor
        Zero momentum green function as function of t, where t runs from 0 to
            length - 1
        Tensor of size (bootstrap_n_samples + 1, training_geometry.length),
        where the 0th element is the estimate based on the full sample,
        and the others are values computed using bootstrap samples.

    Notes
    -----
    This is \tilde{G}(t, 0) as defined in eq. (23) of
    https://arxiv.org/pdf/1904.12072.pdf (defined as mean instead of sum over
    spacial directions) and with momentum explicitly set to zero.

    """
    g_func_zeromom = []
    for t in range(training_geometry.length):
        g_tilde_t = torch.zeros(bootstrap_n_samples + 1, dtype=torch.float)
        for x in range(training_geometry.length):
            g_tilde_t += two_point_function(t, x)
        g_func_zeromom.append(g_tilde_t / training_geometry.length)

    return torch.stack(g_func_zeromom).transpose(0, 1)


def effective_pole_mass(zero_momentum_2pf, bootstrap_n_samples=100):
    r"""Calculate the effective pole mass m^eff(t) defined as

        m^eff(t) = arcosh(
            (\tilde{G}(t-1, 0) + \tilde{G}(t+1, 0)) / (2 * \tilde{G}(t, 0))
        )

    from t = 1 to t = L-2, where L is the length of lattice side

    Returns
    -------
    m_t: torch.Tensor
        effective pole mass as a function of t
        Tensor of size (bootstrap_n_samples + 1, training_geometry.length - 2),
        where the 0th element is the estimate based on the full sample,
        and the others are values computed using bootstrap samples.

    Notes
    -----
    This is m^eff(t) as defined in eq. (28) of
    https://arxiv.org/pdf/1904.12072.pdf

    """
    g_func_zeromom = zero_momentum_2pf
    m_t = []
    for i in range(1, g_func_zeromom.size(1) - 1):
        argument = (g_func_zeromom[:, i - 1] + g_func_zeromom[:, i + 1]) / (
            2 * g_func_zeromom[:, i]
        )
        m_t.append(arcosh(argument))

    return torch.stack(m_t).transpose(0, 1)


def susceptibility(training_geometry, two_point_function, bootstrap_n_samples=100):
    r"""Calculate the susceptibility, which is the sum of two point connected
    green functions over all seperations

        \chi = sum_x G(x)

    Returns
    -------
    chi: torch.Tensor
        value for the susceptibility
        Tensor of size (bootstrap_n_samples + 1),
        where the 0th element is the estimate based on the full sample,
        and the others are values computed using bootstrap samples.

    Notes
    -----
    as defined in eq. (25) of https://arxiv.org/pdf/1904.12072.pdf

    """
    chi = torch.zeros(bootstrap_n_samples + 1, dtype=torch.float)
    for t in range(training_geometry.length):
        for x in range(training_geometry.length):
            chi += two_point_function(t, x)

    return chi


def ising_energy(two_point_function, bootstrap_n_samples=100):
    r"""Ising energy defined as

        E = 1/d sum_{\mu} G(\mu)

    where \mu is the possible unit shifts for each dimension: (1, 0) and (0, 1)
    in 2D

    Returns
    -------
    E: torch.Tensor
        value for the Ising energy
        Tensor of size (bootstrap_n_samples + 1),
        where the 0th element is the estimate based on the full sample,
        and the others are values computed using bootstrap samples.

    Notes
    -----
    as defined in eq. (26) of https://arxiv.org/pdf/1904.12072.pdf

    """
    E = (two_point_function(1, 0) + two_point_function(0, 1)) / 2
    return E


#################################
###       Bootstrapping       ###
#################################
class Bootstrap:
    """Return the standard deviation for an observable based on a number of
    'bootstrap' samples."""

    def __init__(self, bootstrap_n_samples):
        self.n_samples = bootstrap_n_samples

    def __call__(self, observable):
        obs_full = observable[0]
        obs_bootstrap = observable[1:]

        variance = torch.mean((obs_bootstrap - obs_full) ** 2, axis=0)
        #bias = torch.mean(obs_bootstrap) - obs_full  # not sure whether to use this

        return variance.sqrt()


def bootstrap(bootstrap_n_samples=100):
    return Bootstrap(bootstrap_n_samples)


###############################
###     Autocorrelation     ###
###############################
def autocorrelation_2pf(training_geometry, volume_averaged_2pf, window=2.0):
    r"""Computes the autocorrelation of the volume-averaged two point function,
    the integrated autocorrelation time, and two other functions related to the
    computation of an optimal window size for the integrated autocorrelation.

    Autocorrelation is defined by

        \Gamma(t) = <G(k)G(k+t)> - <G(k)><G(k+t)>

    where G(k) is the volume-averaged two point function at Monte Carlo timestep 'k',
    and <> represents an average over all timesteps.

    Integrated autocorrelation is defined, for some window size 'W' by

        \tau_{int}(W) = 0.5 + sum_t^W \Gamma(t)

    Exponential autocorrelation is estimated, up to a factor of S as

        S / \tau_{exp}(W) = log( (2\tau_int(W) + 1) / (2\tau_int(W) - 1) )

    The "g" function has a minimum at 'W_opt' where the sum of the statistical error and the
    systematic error due to truncation, in \tau_{int}, has a minimum.

        g(W) = exp( -W / \tau_{exp}(W) ) - \tau_{exp}(W) / \sqrt(W*N)

    The automatic windowing procedure and definitions of \tau_{exp}(W) and g(W)
    are found in section 3.3 of Ulli Wolff: Monte Carlo errors with less errors -
    https://arxiv.org/pdf/hep-lat/0306017.pdf

    Returns
    -------
    autocorrelation:    numpy.array
    tau_int_W:          numpy.array
    tau_exp_W:          numpy.array
    g_W:                numpy.array
    W_opt:              int         - minimum of g_W

    All numpy arrays are truncated at a point 4*W_opt for the sake of plotting.
    """
    x = t = 0  # Should really look at more than one separation
    va_2pf = volume_averaged_2pf(x, t)
    va_2pf -= va_2pf.mean()
    autocorrelation = correlate(va_2pf, va_2pf, mode="same")  # converts to numpy array
    c = np.argmax(autocorrelation)
    autocorrelation = autocorrelation[c:] / autocorrelation[c]

    n_states = va_2pf.size(0)
    tau_int_W = 0.5 + np.cumsum(autocorrelation[1:])
    valid = np.where(tau_int_W > 0.5)[0]
    tau_exp_W = np.ones(tau_int_W.size) * 0.00001  # to prevent domain error in log

    S = window  # read from runcard parameter
    tau_exp_W[valid] = S / (
        np.log((2 * tau_int_W[valid] + 1) / (2 * tau_int_W[valid] - 1))
    )
    W = np.arange(1, tau_int_W.size + 1)
    g_W = np.exp(-W / tau_exp_W) - tau_exp_W / np.sqrt(W * n_states)

    W_opt = np.where(g_W < 0)[0][0]
    w = 4 * W_opt  # where to cut the plot off

    return autocorrelation[:w], tau_int_W[:w], tau_exp_W[:w], g_W[:w], W_opt
