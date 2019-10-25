from tqdm import tqdm
import sys
import time
from math import ceil

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn

from norm_flow_pytorch import NormalisingFlow, shifted_kl

random = np.random.RandomState(1234)
L = 6
N_UNITS = L**2
m_sq, l = -4, 6.975
N_BATCH = 2000

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

action = PhiFourAction(L, m_sq, l)

def split_transformation(length, a_left=True):
    r"""Given a flattened 2D state, represented by a vector \phi, returns a
    matrix transformation, M, which acting on the flattened state seperates
    two halves of the matrix according to a checkerboard pattern into
    \phi = (\phi_a, \phi_b) (by default even sites sent left, odd sites sent
    right). This behaviour can be changed with `a_left` flag.

    """
    if a_left:
        condition = (1, 0)
    else:
        condition = (0, 1)
    N = length**2
    state = np.zeros((length, length)) # define a checkerboard
    state[1::2, 1::2] = 1
    state[::2, ::2] = 1 # even sites are = 1
    flat = state.flatten()
    left = np.zeros((N, N), dtype=np.float32)
    right = np.zeros((N, N), dtype=np.float32)
    # ceil lets this handle length odd, unneccesary for our project
    left[np.arange(np.ceil(N/2), dtype=int), np.where(flat == condition[0])[0]] = 1.
    right[np.arange(np.ceil(N/2), N, dtype=int), np.where(flat == condition[1])[0]] = 1.
    return torch.from_numpy(left), torch.from_numpy(right)

def train(model, epochs, a_left, b_right):
    """example of training loop of model"""
    # create your optimizer and a scheduler
    optimizer = optim.Adadelta(model.parameters(), lr=1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500)
    # let's use tqdm to see progress
    pbar = tqdm(range(epochs), desc=f"loss: N/A")
    for i in pbar:
        # gen simple states
        z = torch.randn((N_BATCH, N_UNITS))
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

def sample(model, a_left, b_right, n_large, target_length):
    r"""
    Sample using Metroplis-Hastings algorithm from a large number of phi configurations.
    We calculate an A = min(1, \frac{\tilde p(phi^i)}{p(phi^i)} \frac{p(phi^j)}{\tilde p(phi^j)})
    Where i is the index of the current phi, and j is the index of the update proposal phi.
    We then generate a random number u, and if u <= A then we accept the update and i = j for the
    next proposal, phi^i is added to the chain of samples, and a new j is picked. If the update
    is rejected, then i=i for the next proposal and a new j is picked. We continue until the chain
    has the desired length.
    """

    with torch.no_grad(): # don't want gradients being tracked in sampling stage
        z = torch.randn((n_large, N_UNITS)) # random z configurations
        phi = model.inverse_map(z) # map using trained model to phi
        log_ptilde = model.forward(phi) # log of probabilities of generated phis using trained model
        S = action(phi)
        chain_len = 0 # intialise current chain length
        sample_distribution = torch.Tensor(target_length, N_UNITS) # intialise tensor to store samples
        accepted, rejected = 0,0 # track accept/reject statistics
        i = np.random.randint(n_large) # random index to start sampling from configurations tensor
        used = [] # track which configurations have been added to the chain
        used.append(i) 
        while chain_len < target_length:
            j = np.random.randint(n_large) # random initial phi^j for update proposal
            while j in used: # make sure we don't pick a phi we have already used
                j = np.random.randint(n_large)
            exponent = log_ptilde[i] - S[j] - log_ptilde[j] + S[i]
            P_accept = np.exp(float(exponent)) # much faster if you tell it to use a float
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
    
    # set seed, hopefully result is reproducible
    torch.manual_seed(0)
    a_left, b_right = split_transformation(L)
    # define simple mode, each network is single layered
    assert (len(sys.argv) == 3) and (sys.argv[1] in ['train', 'load']),\
    'Pass "train" and a model name to train new model or "load" and model name to load existing model'
    if sys.argv[1] == 'train':
        model = NormalisingFlow(
            size_in=N_UNITS, n_affine=8, affine_hidden_shape=(32,)
        )
        epochs = 4000 # Gives a decent enough approx.
        # model needs to learn rotation and rescale
        train(model, epochs, a_left, b_right)
        torch.save(model, 'models/'+sys.argv[2])
    elif sys.argv[1] == 'load':
        model = torch.load('models/'+sys.argv[2])
    target_length = 10000 # Number of length L^2 samples we want
    n_large = 2 * target_length # Number of configurations to generate to sample from. 10*target_length seems to generate enough.
    start_time = time.time() 
    # Perform Metroplis-Hastings sampling
    sample_dist = sample(model, a_left, b_right, n_large, target_length)
    print('Generated phi distribution:')
    print(sample_dist)
    print("Time to run MC for a chain of %s samples on an L=%s lattice: %s seconds" % (target_length, L, (time.time() - start_time)))

if __name__ == "__main__":
    main()
