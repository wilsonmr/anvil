"""
sample.py

Module containing functions related to sampling from a trained model
"""

from math import exp, isfinite, ceil
import logging
from random import random
import torch

from tqdm import tqdm

import numpy as np

from reportengine import collect

log = logging.getLogger(__name__)

# TODO: check available memory?
BATCH_SIZE = 10000


class LogRatioNanError(Exception):
    pass


def gen_candidates(model, base, target, num):
    z, base_log_density = base(num)

    negative_mag = (z.sum(dim=1).sign() < 0).nonzero().squeeze()

    phi, model_log_density = model(z, base_log_density, negative_mag)

    log_ratio = model_log_density - target.log_density(phi)

    if not isfinite(exp(float(min(log_ratio) - max(log_ratio)))):
        raise LogRatioNanError(
            "could run into nans based on minimum and maximum log of ratio of probabilities"
        )

    return phi, log_ratio


def calc_tau_chain(history):
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


def metropolis_test(current_log_ratio, proposal_log_ratio):
    return min(1, exp(float(current_log_ratio - proposal_log_ratio)))


@torch.no_grad()
def metropolis_hastings(
    loaded_model,
    base_dist,
    target_dist,
    sample_size,
    thermalization,
    sample_interval,
):

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
        log.info(f"Integrated autocorrelation time from preliminary sampling phase: {tau:.2g}")
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
    return _metropolis_hastings[0][0]


def tau_chain(_metropolis_hastings):
    return _metropolis_hastings[0][1]


def acceptance(_metropolis_hastings):
    return _metropolis_hastings[0][2]
