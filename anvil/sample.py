"""
sample.py

Module containing functions related to sampling from a trained model
"""

from math import exp, isfinite, ceil, floor

import numpy as np
import torch

from reportengine import collect

from time import time


def chain_autocorrelation(history: torch.Tensor) -> float:
    """Compute the integrated autocorrelation time for the Markov chain.

        \tau_int = 0.5 + sum_{\tau=1}^{\tau_max} \rho(\tau)/\rho(0)

    where \rho(\tau)/\rho(0) is the probability of \tau consecutive rejections,
    which we estimate by

        \rho(\tau)/\rho(0) = # consecutive runs of \tau rejections / (N - \tau)

    See eqs. (16) and (19) in https://arxiv.org/pdf/1904.12072.pdf

    Parameters
    ----------
    history: torch.Tensor
        A 1xN boolean tensor encoding the acceptance/rejection history of the
        Markov chain. An accepted (rejected) move is denoted by True (False).

    Returns
    -------
    integrated_autocorrelation: float
        The integrated autocorrelation time for the Markov chain.

    """

    start_time = time()

    N = len(history)
    autocorrelations = torch.zeros(N-1, dtype=torch.float)
    consecutive_rejections = 0
    
    for step in history:
        if step == True: # move accepted
            if consecutive_rejections > 0: # faster than unnecessarily accessing array
                autocorrelations[1:consecutive_rejections+1] += 1
            consecutive_rejections = 0
        else: # move rejected
            consecutive_rejections += 1
    if consecutive_rejections > 0: # pick up last rejection run
        autocorrelations[1:consecutive_rejections+1] += 1
    
    # Compute integrated autocorrelation
    integrated_autocorrelation = 0.5 + \
        torch.sum(autocorrelations / (N - torch.arange(N - 1)))
    
    time_taken = time() - start_time
    #print(f"time to compute integrated autocorrelation: {time_taken:.2g}")
    
    return integrated_autocorrelation


def produce_samples(loaded_model, action, length, state_i=None, log_ratio_i=None, first=False):

    with torch.no_grad():  # don't track gradients
        z = torch.randn(
            (length, loaded_model.size_in)
        )  # random z configurations
        phi = loaded_model.inverse_map(z)  # map using trained loaded_model to phi
        log_ptilde = loaded_model(phi)
    history = torch.zeros(length, dtype=torch.bool) # accept/reject history
    chain_indices = torch.zeros(length, dtype=torch.long)

    log_ratio = log_ptilde + action(phi)
    if not isfinite(exp(float(min(log_ratio) - max(log_ratio)))):
        raise ValueError("could run into nans")

    if first:  # first batch
        log_ratio_i = log_ratio[0]
        state_i = phi[0]
    else:
        log_ratio[0] = log_ratio_i  # else set first to be last accepted
        phi[0] = state_i
    i = 0
    for j in range(1, length):
        condition = min(1, exp(float(log_ratio_i - log_ratio[j])))
        if np.random.uniform() <= condition:
            chain_indices[j] = j
            log_ratio_i = log_ratio[j]
            state_i = phi[j]
            i = j
            history[j] = True # accepted -> True
        else:
            chain_indices[j] = i
    
    return phi, chain_indices, history, state_i, log_ratio_i


def sample(loaded_model, action, target_length: int, tskip_guess=-1) -> torch.Tensor:
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

    Returns
    -------
    sample: torch.Tensor
        a sample of states from loaded_model, size = (target_length, loaded_model.size_in)

    """
    batch_length = min(target_length, 10000) # hard coded for now
    
    # Initial batch, from which the autocorrelation time can be estimated
    phi, chain_indices, batch_history, state_i, log_ratio_i = produce_samples(
        loaded_model, action, batch_length, first=True)
    
    if tskip_guess < 1: # i.e. no guess has been attempted
        tskip_guess = ceil(2 * chain_autocorrelation(batch_history))
        print(f"Guess for tskip: {tskip_guess}, based on {batch_length} configurations.")

    # Decide how many configurations to generate, in order to get approximately
    # target_length after picking out decorrelated configurations
    n_large = target_length * tskip_guess
    Nbatches = ceil(n_large / batch_length)
    n_large = Nbatches * batch_length
    
    full_chain = torch.empty((n_large, loaded_model.size_in), dtype=torch.float32)
    history = torch.empty(n_large, dtype=torch.bool) # accept/reject history
    
    # Add first batch
    full_chain[:batch_length, :] = phi[chain_indices, :]
    history[:batch_length] = batch_history

    for batch in range(1, Nbatches):
        # Generate sub-chain of batch_length configurations
        phi, chain_indices, batch_history, state_i, log_ratio_i = produce_samples(
            loaded_model, action, batch_length, state_i, log_ratio_i)
        
        # Add to larger chain
        start = batch*batch_length
        full_chain[start:start+batch_length, :] = phi[chain_indices, :]
        history[start:start+batch_length] = batch_history
    
    # Accept-reject statistics
    accepted = torch.sum(history)
    rejected = batch*batch_length - accepted
    fraction = accepted / float(accepted + rejected)
    print(
        f"Accepted: {accepted}, Rejected: {rejected}, Fraction: "
        f"{fraction:.2g}"
    )

    # Pick out every tskip configuration, which should be decorrelated
    tau_int = chain_autocorrelation(history)
    tskip = ceil(2 * tau_int)
    decorrelated_chain = full_chain[tskip::tskip] # allow for some thermalisation

    #print(f"Integrated autocorrelation time: {tau_int:.2g}")
    print(
        f"Produced a chain of length: {n_large}, but returning a "
        f"decorrelated chain of length: {len(decorrelated_chain)}"
    )
    
    return full_chain


sample_training_output = collect("sample", ("training_context",))
