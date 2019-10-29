"""
sample.py

Run python3 sample.py <model_name> where <model_name> is a pre-trained model saved in a
directory names 'models/'.
"""
import sys
import time
from math import ceil, exp
import random

from tqdm import tqdm
import numpy as np

import torch

from norm_flow_pytorch import NormalisingFlow, shifted_kl
from params import *

def autocorr(acc_rej_list: list, tau: int) -> float:
    """Compute the autocorrelation time for the Markov chain with a given
    time interval tau.

    See Eq. (16)
    """
    
    acc_rej_np = np.array(acc_rej_list[i_therm:], dtype=np.int)
    N = len(acc_rej_np)

    sum_j = 0
    for j in range(N-tau):
        # Term only contributes if all tau proposals were rejected
        sum_j += int(np.all( acc_rej_np[j:j+tau] ))
    
    return sum_j / (N-tau)


# set seed, hopefully result is reproducible
random.seed(1234)
torch.manual_seed(0)
   
model = NormalisingFlow(
    size_in=D, n_affine=n_affine,
    affine_hidden_shape=affine_hidden_shape)
    
model.load_state_dict(torch.load('models/'+sys.argv[1]))
    
# Perform Metroplis-Hastings sampling
start_time = time.time()


with torch.no_grad(): # don't want gradients being tracked in sampling stage
    z = torch.randn((n_large, D)) # random z configurations
    phi = model.inverse_map(z) # map using trained model to phi
    log_p_tilde = model.forward(phi) # log of probabilities of generated phis using trained model
    S = action(phi)
    
# -------------------------------------------------- #
#  Construct Markov Chain using Metropolis-Hastings  #
# -------------------------------------------------- #
    chain = torch.zeros(target_length, D)
    
    N_states = phi.shape[0]
    states_remaining = list(range(N_states))
    
    # Pick a starting configuration, but don't add it to the chain
    i = random.choice(states_remaining)
    
    # List to record series of acceptions/rejections
    acc_rej = []

    chain_len = 0
    while chain_len < target_length:
        j = random.choice(states_remaining)
        
        expnt = min(0, log_p_tilde[i] + S[i] - log_p_tilde[j] - S[j])
        P_accept = exp(float(expnt))

        u = random.random() 
        if u <= P_accept:
            chain[chain_len,:] = phi[j,:]
            chain_len += 1
            acc_rej.append(False)       # False (0) for accepted moves!
            states_remaining.remove(j)
            i = j
        else:
            acc_rej.append(True)


# How did we do?
N_proposals = len(acc_rej)
N_rejected = sum(acc_rej)
N_accepted = N_proposals - N_rejected

print( f"Accepted: {N_accepted}, Rejected: {N_rejected}" )
print( f"Fraction accepted: {N_accepted/N_proposals}" )

# Save state for computing observables
torch.save(phi, "sample_dist.pt")

# --------------- #
# Autocorrelation #
# --------------- #
autocorr_np = np.zeros(tau_max)
for tau in range(tau_max):
    autocorr_np[tau] = autocorr(acc_rej, tau)
np.savetxt("autocorr.txt", autocorr_np)
integrated_autocorr = autocorr_np.sum() + 0.5
# print(integrated_autocorr)

    
print('Generated phi distribution:')
print(chain)
print(
    f"Time to run MC for a chain of {target_length} "
    f"samples on an L={L} lattice: {time.time() - start_time} seconds"
)
