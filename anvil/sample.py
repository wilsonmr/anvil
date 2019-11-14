"""
sample.py

Module containing functions related to sampling from a trained model
"""

from math import exp, isfinite, ceil

import numpy as np
import torch

from reportengine import collect


def sample(loaded_model, action, target_length: int, n_large=20000) -> torch.Tensor:
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
    loaded_model: Module
        loaded_model which is going to be used to generate sample states
    action: Module
        the action upon which the loaded_model was trained, used to calculate the
        acceptance condition
    target_length: int
        the desired number of states to generate from the loaded_model
    n_large: int (default 20000)
        the number of total states to generate from the loaded_model per batch, set
        to a reasonable default value. setting this number excessively high can
        cause slow down due to running out of memory

    Returns
    -------
    sample: torch.Tensor
        a sample of states from loaded_model, size = (target_length, loaded_model.size_in)

    """
    accepted = 0
    rejected = 0
    batches = torch.empty((target_length, loaded_model.size_in), dtype=torch.float32)

    for batch in range(ceil(target_length/n_large)):
        batch_length = min(n_large, target_length-n_large*batch)

        with torch.no_grad():  # don't track gradients
            z = torch.randn((batch_length, loaded_model.size_in))  # random z configurations
            phi = loaded_model.inverse_map(z)  # map using trained loaded_model to phi
            log_ptilde = loaded_model(phi)
        accept_reject = torch.zeros(batch_length, dtype=bool)
        chain_indices = torch.zeros(batch_length, dtype=torch.long)

        log_ratio = log_ptilde + action(phi)
        if not isfinite(exp(float(min(log_ratio) - max(log_ratio)))):
            raise ValueError("could run into nans")

        if batch == 0:  # first batch
            log_ratio_i = log_ratio[0]
            state_i = phi[0]
        else:
            log_ratio[0] = log_ratio_i # else set first to be last accepted
            phi[0] = state_i
        i = 0
        for j in range(1, batch_length):
            condition = min(1, exp(float(log_ratio_i - log_ratio[j])))
            if np.random.uniform() <= condition:
                accept_reject[j] = True
                chain_indices[j] = j
                log_ratio_i = log_ratio[j]
                state_i = phi[j]
                accepted += 1
                i = j
            else:
                accept_reject[j] = False
                chain_indices[j] = i
                rejected += 1
        start = n_large*batch
        end = start + batch_length
        batches[start:end, :] = phi[chain_indices, :]

    print(
        f"Accepted: {accepted}, Rejected: {rejected}, Fraction: "
        f"{accepted/(accepted+rejected):.2g}"
    )
    return batches  # join up batches


sample_training_output = collect("sample", ("training_context",))
