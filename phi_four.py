from tqdm import tqdm
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from norm_flow_pytorch import NormalisingFlow, shifted_kl

class array_mapping(object):

    def __init__(self, array):
        self.array = array
        assert array.shape[0]==array.shape[1], 'Lattice should be square'
        assert array.shape[0]%2==0, 'Lattice should have even number of points per side'
        self.L = array.shape[0]
        self.flattened = self.flatten_checkerboard(self.array)
        self.indices_array()
        self.nn_indices()
        self.nn_arrays = np.array([self.nns_1, self.nns_2, self.nns_3, self.nns_4])
        self.get_nn_val_array()

    @staticmethod
    def flatten_checkerboard(array):
        # Takes LxL array and returns flattened 1x2L array in 
        # checkerboard pattern.
        a, b = [], []
        y, x = array.shape[0], array.shape[1]
        for index, x in np.ndenumerate(array):
            if (index[0]+index[1])%2 == 0:
                a.append(x)
            else:
                b.append(x)
        flattened = np.hstack((a,b))
        return flattened

    def indices_array(self):
        # Creates array in same shape as original array where
        # each element gives the index in the flattened array of the
        # object at that position
        indices = np.zeros((self.L, self.L))
        for index, x in np.ndenumerate(self.array):
            indices[index] = (self.L*index[0] +index[1] - index[1]%2)/2
            if (index[0]+index[1])%2!=0:
                indices[index]+=self.L**2 /2
        self.flat_indices = indices.astype(int)

    def nn_indices(self):
        # Each array here gives nearest neighbours in the given direction, 1-4.
        # The object at (i,j) is the index in the flat array of the nearest neighbour
        # to the object at (i,j) in the input array.
        self.nns_1 = self.flatten_checkerboard(np.roll(self.flat_indices, -1, axis=1))
        self.nns_2 = self.flatten_checkerboard(np.roll(self.flat_indices, 1, axis=1))
        self.nns_3 = self.flatten_checkerboard(np.roll(self.flat_indices, -1, axis=0))
        self.nns_4 = self.flatten_checkerboard(np.roll(self.flat_indices, 1, axis=0))
    
    def get_nn_val_array(self):
        # Create array. Each row is nns in direction [row]+1.
        # Each entry is the value of the nearest neighbour in given direction
        # for entry in flat array in that position.
        self.nn_vals = np.zeros((4, self.L**2))
        for k in range(4):
            for l in range(self.L**2):
                index = self.nn_arrays[k, l]
                self.nn_vals[k,l] = self.flattened[index]

    def get_nns_by_flat_index(self, index, directions):
        assert type(directions)==list, 'Directions parameter must be a list'
        assert all(d>0 and d<5 and type(d)==int for d in directions), 'All directions must be integers in range 1-4'
        nns = []
        for direction in directions:
            nns.append(self.nn_vals[direction-1, index])
        return nns

    @staticmethod
    def split(array, a_left=True):
        half = int(len(array)/2)
        if a_left == True:
            left, right = array[:half], array[half:]
        else:
            right, left = array[:half], array[half:]
        return left, right

random = np.random.RandomState(1234)
L = 6
N_UNITS = L**2
m_sq, l = -4, 6.975
mu, sigma = 0, 1
N_BATCH = 2000
phi_initial_np = (random.normal(mu, sigma, (L, L))).astype(np.float32)
phi_obj = array_mapping(phi_initial_np)
phi = phi_obj.flattened
indices_array = phi_obj.indices_array
shift = phi_obj.nn_arrays
phi = torch.from_numpy(phi)
# A_NP = np.array(np.random.rand(N_UNITS, N_UNITS), dtype=np.float32)
# A = torch.from_numpy(A_NP)
# COV_TARGET = A@A.T
# INV = torch.inverse(COV_TARGET)

def target_distribution_s(phi, m_sq, l, shift):
    r"""Returns action S(\phi) for a stack of states \phi, shape (N_states, D).
    """
    action = 0
    action += (2+0.5*m_sq)*torch.sum(phi**2, dim=1) + l*torch.sum(phi**4, dim=1)
    for i in range(phi.shape[1]):
        phi_x = phi[:,i]
        neighbours_indexes = shift[:,i]
        action -= 0.5*phi_x*torch.sum(phi[:,neighbours_indexes], dim=1)
    action = torch.reshape(action, (phi.shape[0],1))
    return action

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
        target = target_distribution_s(phi, m_sq, l, shift)
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
        # p_tilde = torch.exp(model.forward(phi)) # probabilities of generated phis using trained model
        log_ptilde = model.forward(phi)
        # p = torch.exp(-target_distribution_s(phi, m_sq, l, shift)) # probabilities of phis from target pdf
        S = -target_distribution_s(phi, m_sq, l, shift)
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
            # ratio = (p_tilde[i]/p_tilde[j]) * (p[j]/p[i]) # calculate the ratio in A
            exponent = log_ptilde[i] + S[j] - log_ptilde[j] - S[i]
            P_accept = np.exp(float(exponent)) # much faster if you tell it to use a float
            # A = min(1, ratio) # set A
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
