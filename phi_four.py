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

def sample(model, action, n_large, target_length):
    r"""
    Sample using Metroplis-Hastings algorithm from a large number of phi configurations.
    We calculate an A = min(1, \frac{\tilde p(phi^i)}{p(phi^i)} \frac{p(phi^j)}{\tilde p(phi^j)})
    Where i is the index of the current phi, and j is the index of the update proposal phi.
    We then generate a random number u, and if u <= A then we accept the update and i = j for the
    next proposal, phi^i is added to the chain of samples, and a new j is picked. If the update
    is rejected, then i=i for the next proposal and a new j is picked. We continue until the chain
    has the desired length.
    """
    n_units = model.size_in
    with torch.no_grad(): # don't want gradients being tracked in sampling stage
        z = torch.randn((n_large, n_units)) # random z configurations
        phi = model.inverse_map(z) # map using trained model to phi
        log_ptilde = model(phi) # log of probabilities of generated phis using trained model
        S = action(phi)
        chain_len = 0 # intialise current chain length
        sample_distribution = torch.Tensor(target_length, n_units) # intialise tensor to store samples
        accepted, rejected = 0,0 # track accept/reject statistics
        i = np.random.randint(n_large) # random index to start sampling from configurations tensor
        used = [] # track which configurations have been added to the chain
        used.append(i) 
        while chain_len < target_length:
            j = np.random.randint(n_large) # random initial phi^j for update proposal
            while j in used: # make sure we don't pick a phi we have already used
                j = np.random.randint(n_large)
            exponent = log_ptilde[i] - S[j] - log_ptilde[j] + S[i]
            P_accept = exp(float(exponent)) # much faster if you tell it to use a float
            A = min(1, P_accept) # faster than np.min and torch.min
            u = np.random.uniform() # pick a random u for comparison
            if u <= A:
                sample_distribution[chain_len,:] = phi[i,:] # add to chain if accepted
                chain_len += 1 
                i = j # update i for next proposal
                used.append(i)
                accepted += 1
            else:
                rejected += 1
    print('Accepted: '+str(accepted)+', Rejected:'+str(rejected))
    print('Fraction accepted: '+str(accepted/(accepted+rejected)))
    return sample_distribution

def main():
    length = 6
    n_units = length**2
    m_sq, lam = -4, 6.975
    # set seed, hopefully result is reproducible
    #torch.manual_seed(0)
    action = PhiFourAction(length, m_sq, lam)
    # define simple mode, each network is single layered
    assert (len(sys.argv) == 3) and (sys.argv[1] in ['train', 'load']),\
    'Pass "train" and a model name to train new model or "load" and model name to load existing model'
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
    target_length = 10000 # Number of length L^2 samples we want
    # Number of configurations to generate to sample from.
    n_large = 5*target_length
    start_time = time.time()
    # Perform Metroplis-Hastings sampling
    sample_dist = sample(model, action, n_large, target_length)
    print_plot_observables(sample_dist)
    print(
        f"Time to run MC for a chain of {target_length} "
        f"samples on an L={length} lattice: {time.time() - start_time} seconds"
    )

if __name__ == "__main__":
    main()
