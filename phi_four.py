"""
phi_four.py

script which can train and load models for phi^4 action
"""
import sys
import time
from math import ceil, exp

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from norm_flow_pytorch import NormalisingFlow, shifted_kl

def get_shift(length: int) -> torch.Tensor:
    r"""Given a 2D state of size lengthxlength returns a 4x(length^2)
    tensor where each row gives the 4 nearest neighbours to a flattened state
    which has been split into (\phi_even, \phi_odd) where even/odd refer to
    parity of the site
    """
    # define a checkerboard
    checkerboard = torch.zeros((length, length), dtype=torch.int)
    # set even sites to 1
    checkerboard[1::2, 1::2] = 1
    checkerboard[::2, ::2] = 1

    # make 2d state-like matrix filled with corresponding indices in split-flat state
    splitind_like_state = torch.zeros((length, length), dtype=torch.int)
    splitind_like_state[checkerboard.bool()] = torch.arange(
        int(ceil(length**2/2)),
        dtype=torch.int,
    )
    splitind_like_state[~checkerboard.bool()] = torch.arange(
        int(ceil(length**2/2)),
        length**2,
        dtype=torch.int,
    )

    flat_checker = checkerboard.flatten()
    # make split-flat state like object with corresponding indices in flat state
    out_ind = torch.cat(
        [torch.where(flat_checker == 1)[0], torch.where(flat_checker == 0)[0]],
        dim=0,
    )

    direction_dimension = [
        (-1, 1),
        (1, 1),
        (-1, 0),
        (1, 0)
    ]
    shift = torch.zeros(4, length*length, dtype=torch.long)
    for i, (direction, dim) in enumerate(direction_dimension):
        # each shift, roll the 2d state-like indices and then flatten and split
        shift[i, :] = splitind_like_state.roll(direction, dims=dim).flatten()[out_ind]
    return shift

class PhiFourAction(nn.Module):
    """Extend the nn.Module class to return the phi^4 action given a state
    might be possible to jit compile this to make training a bit faster
    """
    def __init__(self, length, m_sq, lam):
        super(PhiFourAction, self).__init__()
        self.shift = get_shift(length)
        self.lam = lam
        self.m_sq = m_sq
        self.length = length

    def forward(self, phi_state: torch.Tensor) -> torch.Tensor:
        """Given a stack of states, calculate the action for each state"""
        action = (
            (2+0.5*self.m_sq)*phi_state**2 + # phi^2 terms
            self.lam*phi_state**4 - #phi^4 term
            0.5*torch.sum(
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

RNG = np.random.RandomState(1234) # seed MCMC for now

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
        log_ptilde = model.forward(phi) # log of probabilities of generated phis using trained model
        S = action(phi)
        chain_len = 0 # intialise current chain length
        sample_distribution = torch.Tensor(target_length, n_units) # intialise tensor to store samples
        accepted, rejected = 0,0 # track accept/reject statistics
        i = RNG.randint(n_large) # random index to start sampling from configurations tensor
        used = [] # track which configurations have been added to the chain
        used.append(i) 
        while chain_len < target_length:
            j = RNG.randint(n_large) # random initial phi^j for update proposal
            while j in used: # make sure we don't pick a phi we have already used
                j = RNG.randint(n_large)
            exponent = log_ptilde[i] - S[j] - log_ptilde[j] + S[i]
            P_accept = exp(float(exponent)) # much faster if you tell it to use a float
            A = min(1, P_accept) # faster than np.min and torch.min
            u = RNG.uniform() # pick a random u for comparison
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
    torch.manual_seed(0)
    action = PhiFourAction(length, m_sq, lam)
    # define simple mode, each network is single layered
    assert (len(sys.argv) == 3) and (sys.argv[1] in ['train', 'load']),\
    'Pass "train" and a model name to train new model or "load" and model name to load existing model'
    if sys.argv[1] == 'train':
        model = NormalisingFlow(
            size_in=n_units, n_affine=8, affine_hidden_shape=(32,)
        )
        epochs = 4000 # Gives a decent enough approx.
        # model needs to learn rotation and rescale
        start_train_time = time.time()
        train(model, action, epochs)
        torch.save(model.state_dict(), 'models/'+sys.argv[2])
    elif sys.argv[1] == 'load':
        model = NormalisingFlow(
            size_in=n_units, n_affine=8, affine_hidden_shape=(32,)
        )
        model.load_state_dict(torch.load('models/'+sys.argv[2]))
    target_length = 10000 # Number of length L^2 samples we want
    # Number of configurations to generate to sample from.
    n_large = 2*target_length
    start_time = time.time()
    # Perform Metroplis-Hastings sampling
    sample_dist = sample(model, action, n_large, target_length)
    print('Generated phi distribution:')
    print(sample_dist)
    print(
        f"Time to run MC for a chain of {target_length} "
        f"samples on an L={length} lattice: {time.time() - start_time} seconds"
    )

if __name__ == "__main__":
    main()
