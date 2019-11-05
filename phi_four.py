"""
phi_four.py

script which can train and load models for phi^4 action
"""
import sys
import time
from math import exp

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from norm_flow_pytorch import NormalisingFlow, shifted_kl
from geometry import get_shift
from observables import print_plot_observables

class PhiFourAction(nn.Module):
    """Extend the nn.Module class to return the phi^4 action given either
    a single state size (1, length * length) or a stack of N states
    (N, length * length). See Notes about action definition.

    Parameters
    ----------
    length: int
        defines 2D lattice size (length * length)
    m_sq: float
        the value of the bare mass squared
    lam: float
        the value of the bare coupling

    Examples
    --------
    Consider the toy example of the action acting on a random state

    >>> action = PhiFourAction(2, 1, 1)
    >>> state = torch.rand((1, 2*2))
    >>> action(state)
    tensor([[0.9138]])

    Now consider a stack of states

    >>> stack_of_states = torch.rand((5, 2*2))
    >>> action(stack_of_states)
    tensor([[3.7782],
            [2.8707],
            [4.0511],
            [2.2342],
            [2.6494]])

    Notes
    -----
    that this is the action as defined in
    https://doi.org/10.1103/PhysRevD.100.034515 which might differ from the
    current version on the arxiv.

    """
    def __init__(self, length, m_sq, lam):
        super(PhiFourAction, self).__init__()
        self.shift = get_shift(length)
        self.lam = lam
        self.m_sq = m_sq
        self.length = length

    def forward(self, phi_state: torch.Tensor) -> torch.Tensor:
        """Perform forward pass, returning action for stack of states.

        see class Notes for details on definition of action.
        """
        action = (
            (2+0.5*self.m_sq)*phi_state**2 + # phi^2 terms
            self.lam*phi_state**4 - #phi^4 term
            torch.sum(
                phi_state[:, self.shift]*phi_state.view(-1, 1, self.length**2),
                dim=1,
            ) # derivative
        ).sum(dim=1, keepdim=True) # sum across sites
        return action

N_BATCH = 2000 # keep batch size constant for now

def train(model, action, epochs):
    """example of training loop of model"""
    # create your optimizer and a scheduler
    optimizer = optim.Adadelta(model.parameters(), lr=1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500)
    # let's use tqdm to see progress
    pbar = tqdm(range(epochs), desc=f"loss: N/A")
    n_units = model.size_in
    for i in pbar:
        # gen simple states
        z = torch.randn((N_BATCH, n_units))
        phi = model.inverse_map(z)
        target = action(phi)
        output = model(phi)

        model.zero_grad() # get rid of stored gradients
        loss = shifted_kl(output, target)
        loss.backward() # calc gradients

        optimizer.step()
        scheduler.step(loss)

        if (i%50) == 0:
            pbar.set_description(f"loss: {loss.item()}")

def sample(
        model,
        action,
        target_length: int,
        n_large=20000
    ) -> torch.Tensor:
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
        with torch.no_grad(): # don't track gradients
            z = torch.randn((n_large, model.size_in)) # random z configurations
            phi = model.inverse_map(z) # map using trained model to phi
            log_ptilde = model(phi)
        accept_reject = torch.zeros(n_large, dtype=bool)

        log_ratio = log_ptilde + action(phi)
        if ~np.isfinite(exp(float(min(log_ratio) - max(log_ratio)))):
            raise ValueError("could run into nans")

        if not batches: # first batch
            log_ratio_i = log_ratio[0]
            start = 1
        else:
            start = 0 # keep last log_ratio_i and start from 0

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
    return torch.cat(batches, dim=0) # join up batches

class InvalidArgsError(Exception): pass

def main():
    """main loop for phi_four.py"""
    length = 6
    n_units = length**2
    m_sq, lam = -4, 6.975
    # set seed, hopefully result is reproducible
    #torch.manual_seed(0)
    action = PhiFourAction(length, m_sq, lam)
    # define simple mode, each network is single layered
    if not ((len(sys.argv) == 3) and (sys.argv[1] in ['train', 'load'])):
        raise InvalidArgsError(
            "Pass `train` and a model name to train new model or "
            "`load` and model name to load existing model"
        )

    if sys.argv[1] == 'train':
        model = NormalisingFlow(
            size_in=n_units, n_affine=8, affine_hidden_shape=(32,)
        )
        epochs = 10000 # Gives a decent enough approx.
        train(model, action, epochs)
        torch.save(model.state_dict(), 'models/'+sys.argv[2])
    elif sys.argv[1] == 'load':
        model = NormalisingFlow(
            size_in=n_units, n_affine=8, affine_hidden_shape=(32, )
        )
        model.load_state_dict(torch.load('models/'+sys.argv[2]))
    target_length = 100000 # Number of samples we want

    start_time = time.time()
    # Perform Metroplis-Hastings sampling
    sample_dist = sample(model, action, target_length)
    print_plot_observables(sample_dist)
    print(
        f"Time to run MC for a chain of {target_length} "
        f"samples on an L={length} lattice: {time.time() - start_time} seconds"
    )

if __name__ == "__main__":
    main()
