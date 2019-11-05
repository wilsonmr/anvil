"""
sample.py

Module containing functions related to sampling from a trained model
"""

from math import exp, isfinite

import numpy as np
import torch


def sample(model, action, target_length: int, n_large=20000) -> torch.Tensor:
    r"""
    Sample using Metroplis-Hastings algorithm from a large number of phi
    configurations.

    We calculate the condition

        A = min[1, (\tilde p(phi^i) * p(phi^j)) / (p(phi^i) * \tilde p(phi^j))]

    Where i is the index of the current phi in metropolise chain and j is the
    current proposal. A uniform random number, u, is drawn and if u <= A then
    the proposed state phi^j is accepted (and becomes phi^i for the next update)

    Parameters
    ----------
    model: Module
        model which is going to be used to generate sample states
    action: Module
        the action upon which the model was trained, used to calculate the
        acceptance condition
    target_length: int
        the desired number of states to generate from the model
    n_large: int (default 20000)
        the number of total states to generate from the model per batch, set
        to a reasonable default value. setting this number excessively high can
        cause slow down due to running out of memory

    Returns
    -------
    sample: torch.Tensor
        a sample of states from model, size = (target_length, model.size_in)

    """
    accepted = 0
    rejected = 0
    batches = []

    while accepted < target_length:
        with torch.no_grad():  # don't track gradients
            z = torch.randn((n_large, model.size_in))  # random z configurations
            phi = model.inverse_map(z)  # map using trained model to phi
            log_ptilde = model(phi)
        accept_reject = torch.zeros(n_large, dtype=bool)

        log_ratio = log_ptilde + action(phi)
        if not isfinite(exp(float(min(log_ratio) - max(log_ratio)))):
            raise ValueError("could run into nans")

        if not batches:  # first batch
            log_ratio_i = log_ratio[0]
            start = 1
        else:
            start = 0  # keep last log_ratio_i and start from 0

        for j in range(start, n_large):
            condition = min(1, exp(float(log_ratio_i - log_ratio[j])))
            if np.random.uniform() <= condition:
                accept_reject[j] = True
                log_ratio_i = log_ratio[j]
                accepted += 1
                if accepted == target_length:
                    break
            else:
                accept_reject[j] = False
                rejected += 1
        batches.append(phi[accept_reject, :])

    print(
        f"Accepted: {accepted}, Rejected: {rejected}, Fraction: "
        f"{accepted/(accepted+rejected):.2g}"
    )
    return torch.cat(batches, dim=0)  # join up batches
