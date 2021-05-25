# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
sample.py

Module containing functions related to sampling from a trained model
"""
from math import exp, isfinite, ceil
import logging
from random import random
from tqdm import tqdm

import numpy as np
import torch

from reportengine import collect

log = logging.getLogger(__name__)

# TODO: check available memory?
BATCH_SIZE = 10000


class LogRatioNanError(Exception):
    pass


def gen_candidates(model, base, target, num: int) -> tuple:
    r"""Generates a batch of candidate field configurations from a trained model.

    Returns the batch along with the 'log ratio'

    .. math::

        \log (\tilde{p}_\theta(\phi)) - \log(p(\phi))

    where the global shift due to the normalisations of the probability
    densities is not calculated (in any case the normalisation of the target
    is typically not known).

    Parameters
    ----------
    model
        Traing flow model which is able to map latent variables from the base
        distribution to candidate field configurations distributed according to
        :math:`\tilde{p}_\theta(\phi)`
    base
        Base distribution from which latent variables are generated.
    target
        Target distribution - probably a field theory - from which we would
        like to sample.
    num
        Number of candidate configurations to generate in this batch.

    Returns
    -------
    torch.Tensor
        Tensor containing the candidate configurations, dimensions
        ``(num, lattice_size)``
    torch.Tensor
        Tensor containing the 'log ratio' as defined above.

    """
    z, base_log_density = base(num)

    negative_mag = (z.sum(dim=1).sign() < 0).nonzero().squeeze()

    phi, model_log_density = model(z, base_log_density, negative_mag)

    log_ratio = model_log_density - target.log_density(phi)

    if not isfinite(exp(float(min(log_ratio) - max(log_ratio)))):
        raise LogRatioNanError(
            "could run into nans based on minimum and maximum log of ratio of probabilities"
        )

    return phi, log_ratio


def calc_tau_chain(history: list) -> float:
    r"""Returns an estimate of the integrated autocorrelation time based on the
    accept/reject 'history' of a Metropolis simulation.

    The estimate is based on the insight that the autocorrelation function is
    given by

    .. math::

        \frac{\Gamma(t)}{\Gamma(0)} =
        \Pr( t \text{ consecutive rejections} )

    which can be easily estimated from a finite sampling phase.

    Parameters
    ----------
    history
        List containing the history of the Mertropolis simulation in the form of
        ones (proposal accepted) and zeros (proposal rejected)

    Returns
    -------
    float
        Estimate of the integrated autocorrelation time.

    Notes
    -----
    Reference: https://arxiv.org/abs/1904.12072

    """
    n_states = len(history)
    autocorrelations = torch.zeros(
        n_states + 1, dtype=torch.float
    )  # +1 in case 100% rejected
    consecutive_rejections = 0

    for step in history:
        if step:  # move accepted
            if consecutive_rejections > 0:  # faster than unnecessarily accessing array
                autocorrelations[1 : consecutive_rejections + 1] += torch.arange(
                    consecutive_rejections, 0, -1, dtype=torch.float
                )
            consecutive_rejections = 0
        else:  # move rejected
            consecutive_rejections += 1
    if consecutive_rejections > 0:  # pick up last rejection run
        autocorrelations[1 : consecutive_rejections + 1] += torch.arange(
            consecutive_rejections, 0, -1, dtype=torch.float
        )

    # Compute integrated autocorrelation
    integrated_autocorrelation = 0.5 + torch.sum(
        autocorrelations / torch.arange(n_states + 1, 0, -1, dtype=torch.float)
    )
    return float(integrated_autocorrelation)


def metropolis_test(current_log_ratio, proposal_log_ratio) -> float:
    r"""Returns the Metropolis acceptance probability for a transition between a
    given current and proposed configuration.

    The Metropolis test is defined by a conditional probability of transitioning
    from the current state :math:`\phi` to a proposed state :math:`\phi'`, where
    the proposal is drawn from the distribution :math:`\tilde{p}_\theta(\phi')`
    and the target distribution is :math:`p(\phi)`:

    .. math::

        A(\phi \to \phi') = \min \left( 1,
        \frac{\tilde{p}_\theta(\phi)}{p(\phi)}
        \frac{p(\phi')}{\tilde{p}_\theta(\phi')} \right)

    Note that normalisations in this expression cancel.

    Parameters
    ----------
    current_log_ratio
        The logarithm of the ratio of model density to target density for the
        current configuration, ignoring normalisations.
    proposal_log_ratio
        The logarithm of the ratio of model density to target density for the
        proposed configuration, ignoring normalisations.

    Returns
    -------
    float
        The probability :math:`A(\phi \to \phi')`.

    Notes
    -----
    It is assumed that the model generates *independent* candidate configurations,
    i.e. :math:`\tilde{p}_\theta(\phi' \mid \phi) \equiv \tilde{p}_\theta(\phi')`

    """
    return min(1, exp(float(current_log_ratio - proposal_log_ratio)))


@torch.no_grad()
def metropolis_hastings(
    loaded_model,
    base_dist,
    target_dist,
    sample_size: int,
    thermalization: (int, type(None)),
    sample_interval: (int, type(None)),
):
    """Runs a Metropolis-Hastings sampling simulation given a trained model and a
    target probability distribution.

    Parameters
    ----------
    loaded_model
        Trained model which generates candidate configurations by passing latent
        variables through its layers.
    base_dist
        Distribution objet that generates latent variables and an associated
        (log) probability density.
    target_dist
        The distribution from which we would like to sample using the Metropolis-
        Hastings algorithm.
    sample_size
        The number of configurations we desire in the resulting sample.
    thermalization
        The number of updates to discard before we begin accumulating configurations
        in the output sample.
    sample_interval
        Optional user-prescribed interval defining the number of updates to discard
        between each configuration added to the output sample.

    Returns
    -------
    torch.Tensor
        Sample of field configurations distributed according to the target.
    float
        Estimate of the integrated autocorrelation time of the simulation based
        on accept-reject statistics.
    float
        Fraction of proposals which were accepted during the simulation.

    """
    # Draw starting configs
    phi, log_ratio = gen_candidates(loaded_model, base_dist, target_dist, num=1)
    current = phi[0]
    current_log_ratio = log_ratio[0]

    # Thermalization phase
    if thermalization is not None:
        for proposal, proposal_log_ratio in zip(
            *gen_candidates(
                loaded_model,
                base_dist,
                target_dist,
                num=thermalization,
            )
        ):
            if random() < metropolis_test(current_log_ratio, proposal_log_ratio):
                current, current_log_ratio = proposal, proposal_log_ratio

    # Sample interval
    if sample_interval is None:
        history = []
        for proposal, proposal_log_ratio in zip(
            *gen_candidates(
                loaded_model,
                base_dist,
                target_dist,
                num=BATCH_SIZE,
            )
        ):
            if random() < metropolis_test(current_log_ratio, proposal_log_ratio):
                current, current_log_ratio = proposal, proposal_log_ratio
                history.append(1)
            else:
                history.append(0)

        tau = calc_tau_chain(history)
        log.info(
            f"Integrated autocorrelation time from preliminary sampling phase: {tau:.2g}"
        )
        sample_interval = ceil(2 * tau)  # update sample interval

    log.info(f"Using sampling interval: {sample_interval}")

    # Generate representative sample
    configs_out = torch.empty((sample_size, base_dist.size_out), dtype=torch.float32)
    history = []

    batches = [BATCH_SIZE for _ in range((sample_size * sample_interval) // BATCH_SIZE)]
    if (sample_size * sample_interval) % BATCH_SIZE > 0:
        batches.append(sample_size % BATCH_SIZE)

    pbar = tqdm(range(sample_size), desc="configs")
    n = 0
    for batch in batches:
        for proposal, proposal_log_ratio in zip(
            *gen_candidates(
                loaded_model,
                base_dist,
                target_dist,
                num=batch,
            )
        ):
            if random() < metropolis_test(current_log_ratio, proposal_log_ratio):
                current, current_log_ratio = proposal, proposal_log_ratio
                history.append(1)
            else:
                history.append(0)

            if n % sample_interval == 0:
                configs_out[n // sample_interval] = current
                pbar.update(1)

            n += 1

    pbar.close()

    tau = calc_tau_chain(history)
    log.info(f"Integrated autocorrelation time of output sample: {tau:.2g}")

    acceptance = sum(history) / len(history)
    log.info(f"Fraction of proposals accepted: {acceptance:.2g}")

    return configs_out, tau, acceptance


_metropolis_hastings = collect("metropolis_hastings", ("training_context",))


def configs(_metropolis_hastings):
    """Returns sample of configurations from the Metropolis-Hastings simulation."""
    return _metropolis_hastings[0][0].numpy()


def tau_chain(_metropolis_hastings):
    """Returns estimate of the integrated autocorrelation time from the Metropolis-
    Hastings simulation."""
    return _metropolis_hastings[0][1]


def acceptance(_metropolis_hastings):
    """Returns the fraction of proposals that were accepted during the Metropolis-
    Hastings simulation."""
    return _metropolis_hastings[0][2]
