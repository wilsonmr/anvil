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
    # c0 = torch.log(x)
    # c1 = torch.log1p(torch.sqrt(x * x - 1) / x)
    # return c0 + c1
    # NOTE: might need stable version here
    return torch.log(x + torch.sqrt(pow(x, 2) - 1))


def calc_two_point_function(sample_training_output, training_geometry):
    r"""Calculates the two point connected green function, G(x), given a set of
    states where x = (x_0, x_1) refers to a shift applied to the fields \phi

    Parameters
    ----------
    sample_training_output: torch.Tensor
        a stack of phi states sampled from the model. First dimension is the
        `batch dimension` and is length N_states. Second dimension is the
        `lattice dimension` which is of length N_lattice.
    training_geometry:
        a geometry class as defined in geometry.py, used primarily for the
        nearest neighbour shift.

    Returns
    -------
    g_func: torch.Tensor
        2d tensor of size (lattice length, lattice length, *)
        containing values of green function G(x) (allowing for bootstrap
        dimension)
    """
    # allow for additional dimensions like bootstrap
    _, _, *extra_dims = sample_training_output.shape

    g_func = torch.empty(
        (training_geometry.length, training_geometry.length, *extra_dims)
    )
    for i in range(training_geometry.length):
        for j in range(training_geometry.length):
            shift = training_geometry.get_shift(shifts=((i, j),), dims=((0, 1),)).view(
                -1
            )  # make 1d

            # Sample of size (target_length, n_states)
            phi = sample_training_output
            phi_shift = phi[:, shift]

            #  Average over stack of states
            phi_mean = phi.mean(dim=0)
            phi_shift_mean = phi_shift.mean(dim=0)
            phi_shift_phi_mean = (phi_shift * phi).mean(dim=0)

            # Average over coordinates
            g_func[i, j] = torch.mean(
                phi_shift_phi_mean - phi_shift_mean * phi_mean, dim=0
            )
    return g_func


def volume_avg_two_point_function(sample_training_output, training_geometry):
    """
    Return torch Tensor of volume-averaged two point functions, defined like
    in calc_two_point_function except the mean is just taken over lattice sites
    for each shift, no mean is taken across batch dimension

    Parameters
    ----------
    sample_training_output: torch.Tensor
        a stack of phi states sampled from the model. First dimension is the
        `batch dimension` and is length N_states. Second dimension is the
        `lattice dimension` which is of length N_lattice.
    training_geometry:
        a geometry class as defined in geometry.py, used primarily for the
        nearest neighbour shift.

    Returns
    -------
    va_2pf: torch.Tensor
        A 3 dimensional Tensor containing the volume-averaged two point function
        for each state in the sample, for each set of shifts shape
        (N_states, lattice length, lattice length)

    """

    va_2pf = torch.empty_like(sample_training_output).view(
        -1, training_geometry.length, training_geometry.length
    )
    for i in range(training_geometry.length):
        for j in range(training_geometry.length):
            shift = training_geometry.get_shift(shifts=((i, j),), dims=((0, 1),)).view(
                -1
            )

            va_2pf[:, i, j] = (
                sample_training_output[:, shift] * sample_training_output
            ).mean(dim=1) - sample_training_output.mean(dim=1).pow(2)
    return va_2pf


def bootstrap_function(func, states, *args, n_boot=100):
    """Take a func which expects N_batch on the first dimension and can handle
    an extra bootstrap dimension on final dimension and return n_boot resamples
    of the function, with the resamples on the final dimension of the returned
    tensor

    For large samples and/or large n_boot, it can be impossible to allocate
    enough memory for a single tensor containing all bootstrap resamples.
    Furthermore, even if memory can be allocated, it is often faster to
    do the calculations separately on several smaller tensors.

    Parameters
    ----------
    func:
        function which is to be resampled, should be able to take tensor of shape
        (N_states, N_lattice, N_boot)
    states: torch.Tensor
        states of shape (N_states, N_lattice)
    *args:
        other positional arguments of the `func`
    n_boot: int, default 100
        number of resamples, by default set to 100

    Returns
    -------
    resampled_func: torch.Tensor
        with shape (*func(states, *args).shape, n_boot), resamples are on final
        dimension

    """
    sample_size = states.shape[0]

    res = []
    for resample in range(n_boot):
        boot_index = torch.randint(0, sample_size, size=(sample_size,))
        resampled_states = states[boot_index, :]
        res.append(func(resampled_states, *args))

    return torch.stack(res, dim=-1)


def two_point_function(sample_training_output, training_geometry, n_boot=100):
    """Bootstrap calc_two_point_function, using bootstrap_function"""
    return bootstrap_function(
        calc_two_point_function,
        sample_training_output,
        training_geometry,
        n_boot=n_boot,
    )


def zero_momentum_two_point(two_point_function):
    r"""Calculate the zero momentum green function as a function of t
    \tilde{G}(t, 0) which is assumed to be in the first dimension defined as

        \tilde{G}(t, 0) = 1/L \sum_{x_1} G(t, x_1)

    Returns
    -------
    g_func_zeromom: torch.Tensor
        Zero momentum green function as function of t, where t is zero indexed.
        Tensor of size (lattice length, n_boot)

    Notes
    -----
    This is \tilde{G}(t, 0) as defined in eq. (23) of
    https://arxiv.org/pdf/1904.12072.pdf (defined as mean instead of sum over
    spacial directions) and with momentum explicitly set to zero.

    """
    # mean across x
    g_tilde_t = two_point_function.mean(dim=1)
    return g_tilde_t


def effective_pole_mass(zero_momentum_two_point):
    r"""Calculate the effective pole mass m^eff(t) defined as

        m^eff(t) = arcosh(
            (\tilde{G}(t-1, 0) + \tilde{G}(t+1, 0)) / (2 * \tilde{G}(t, 0))
        )

    from t = 1 to t = L-2, where L is the length of lattice side

    Returns
    -------
    m_t: torch.Tensor
        effective pole mass as a function of t
        Tensor of size (lattice length - 2, n_boot),

    Notes
    -----
    This is m^eff(t) as defined in eq. (28) of
    https://arxiv.org/pdf/1904.12072.pdf

    """
    inner_indices = torch.tensor(range(1, zero_momentum_two_point.shape[0] - 1))
    res = arcosh(
        (
            zero_momentum_two_point[inner_indices - 1]
            + zero_momentum_two_point[inner_indices + 1]
        )
        / (2 * zero_momentum_two_point[inner_indices])
    )
    return res


def susceptibility(two_point_function):
    r"""Calculate the susceptibility, which is the sum of two point connected
    green functions over all seperations

        \chi = sum_x G(x)

    Returns
    -------
    chi: torch.Tensor
        value for the susceptibility Tensor of size n_boot

    Notes
    -----
    as defined in eq. (25) of https://arxiv.org/pdf/1904.12072.pdf

    """
    return two_point_function.sum(dim=(0, 1))


def ising_energy(two_point_function):
    r"""Ising energy defined as

        E = 1/d sum_{\mu} G(\mu)

    where \mu is the possible unit shifts for each dimension: (1, 0) and (0, 1)
    in 2D

    Returns
    -------
    E: torch.Tensor
        value for the Ising energy Tensor of size n_boot

    Notes
    -----
    as defined in eq. (26) of https://arxiv.org/pdf/1904.12072.pdf

    """
    return (two_point_function[1, 0] + two_point_function[0, 1]) / 2


def autocorr_two_point(volume_avg_two_point_function, window=2.0):
    r"""Computes the autocorrelation of the volume-averaged two point function,
    the integrated autocorrelation time, and two other functions related to the
    computation of an optimal window size for the integrated autocorrelation.

    Autocorrelation is defined by

        \Gamma(t) = <G(k)G(k+t)> - <G(k)><G(k+t)>

    where G(k) is the volume-averaged two point function at Monte Carlo timestep 'k',
    and <> represents an average over all timesteps.

    -----

    Integrated autocorrelation is defined, for some window size 'W' by

        \tau_{int}(W) = 0.5 + sum_t^W \Gamma(t)

    Exponential autocorrelation is estimated, up to a factor of S as

        S / \tau_{exp}(W) = log( (2\tau_int(W) + 1) / (2\tau_int(W) - 1) )

    The "g" function has a minimum at 'W_opt' where the sum of the statistical
    error and the systematic error due to truncation, in \tau_{int}, has a minimum.

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
    # TODO: look at more than one seperation
    x = t = 0
    va_2pf = volume_avg_two_point_function[:, x, t]
    va_2pf -= va_2pf.mean()
    # converts to numpy array
    autocorrelation = correlate(va_2pf, va_2pf, mode="same")
    c = np.argmax(autocorrelation)
    autocorrelation = autocorrelation[c:] / autocorrelation[c]
    return autocorrelation


def integrated_autocorr_two_point(autocorr_two_point):
    r"""Calculate the integrated autocorrelation of the two point function.

    Integrated autocorrelation is defined, for some window size 'W' by

        \tau_{int}(W) = 0.5 + sum_t^W \Gamma(t)

    """
    return 0.5 + np.cumsum(autocorr_two_point[1:])


def exp_autocorr_two_point(integrated_autocorr_two_point, window=2.0):
    """Calculate the exponential autocorrelation of the two point function.

    Exponential autocorrelation is estimated, up to a factor of S as

        S / \tau_{exp}(W) = log( (2\tau_int(W) + 1) / (2\tau_int(W) - 1) )

    """
    tau_int_W = integrated_autocorr_two_point
    valid = np.where(tau_int_W > 0.5)[0]
    tau_exp_W = np.ones(tau_int_W.size) * 0.00001  # to prevent domain error in log

    tau_exp_W[valid] = window / (
        np.log((2 * tau_int_W[valid] + 1) / (2 * tau_int_W[valid] - 1))
    )
    return tau_exp_W


def automatic_windowing_function(
    integrated_autocorr_two_point,
    exp_autocorr_two_point,
    volume_avg_two_point_function,
    window=2.0,
):
    r"""Return the function for estimating optimal window size for integrated
    autocorrelation as defined in equation (52), section 3.3 of
    https://arxiv.org/pdf/hep-lat/0306017.pdf

    The "g" function has a minimum at 'W_opt' where the sum of the statistical
    error and the systematic error due to truncation, in \tau_{int}, has a minimum.

        g(W) = exp( -W / \tau_{exp}(W) ) - \tau_{exp}(W) / \sqrt(W*N)

    """
    n_states = volume_avg_two_point_function.shape[0]
    tau_int = integrated_autocorr_two_point
    tau_exp = exp_autocorr_two_point
    windows = np.arange(1, tau_int.size + 1)
    return np.exp(-windows / tau_exp) - tau_exp / np.sqrt(windows * n_states)


def optimal_window(automatic_windowing_function):
    """using automatic_windowing_function, estimate optimal window, which
    is the first point at which the automatic_windowing_function becomes
    negative
    """
    return np.where(automatic_windowing_function < 0)[0][0]
