"""
train.py

Run python3 train.py <model_name>. The model will be saved as <model_name>.pt in a
directory given by 'model_dir', which is specified in params.py
"""
from sys import argv
import os
import time
from math import ceil, exp
import ast

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim

from norm_flow_pytorch import NormalisingFlow, shifted_kl
from params import *
import sample

# Create model directory if it doesn't exist already
if not os.path.exists(model_dir): os.makedirs(model_dir)
# Create directory for convergence data
if not os.path.exists(training_data_dir): os.makedirs(training_data_dir)

model = NormalisingFlow(
    size_in=D, n_affine=n_affine,
    affine_hidden_shape=affine_hidden_shape)

# Load model if it exists
if os.path.exists(model_dir+argv[1]+'.pt'):
    print("Loading trained model: "+model_dir+argv[1]+'.pt')
    model.load_state_dict(torch.load(model_dir+argv[1]+'.pt'))
    
    # Also want to load convergence data from previous training
    # Throw an error if this doesnt' exist, because it means we don't know how long the
    # previous training stage was run for. Possibly not the best thing to do here...
    assert os.path.exists(training_data_dir+argv[1]+'.txt'),\
            "Couldn't find data from previous training: %s not found." %(training_data_dir+argv[1]+'.txt')

    with open(training_data_dir+argv[1]+'.txt', 'r') as f:
        # Read previous params (as dict) from file header
        params_prev = ast.literal_eval( f.readline().strip()[2:] )
        # Load time taken to reach current state. Needs improvement; poor way to load one number.
        restart_time = float(f.readlines()[-1].strip().split()[0])
    
    # Check we're restarting with the same parameters as before
    for key, val in { 'L':L, 'N_BATCH':N_BATCH, 'n_affine':n_affine,
            'affine_hidden_shape':affine_hidden_shape[0] }.items():
        assert params_prev[key] == val,\
            f"{key}={val} in params.py does not match {key}={params_prev[key]} from previous training"
    
    initial_save = False

# Otherwise start from scratch
else:
    restart_time = 0
    initial_save = True
    header = f"{{ 'L': {L},\
        'N_BATCH': {N_BATCH},\
        'n_affine': {n_affine},\
        'affine_hidden_shape': {affine_hidden_shape[0]},\
        'epochs_sample': {epochs_sample},\
        'target_length': {target_length},\
        'n_large': {n_large} }}"

# ---------- #
#  Training  #
# ---------- #
# create your optimizer and a scheduler
optimizer = optim.Adadelta(model.parameters(), lr=1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500)
# let's use tqdm to see progress
pbar = tqdm(range(epochs), desc=f"loss: N/A")

# set seed, hopefully result is reproducible
torch.manual_seed(0)

# Create numpy array for convergence data
out_array = np.zeros( (epochs//epochs_sample, 4) )
i_out = 0

start_time = time.time()
for i in pbar:
    # gen simple states
    z = torch.randn((N_BATCH, D))
    phi = model.inverse_map(z)
    target = action(phi)
    output = model(phi)

    model.zero_grad() # get rid of stored gradients
    loss = shifted_kl(output, target)
    loss.backward() # calc gradients

    optimizer.step()
    scheduler.step(loss)

    if (i%50) == 0:
        pbar.set_description(f"loss: {loss.item():.4f}")

    if ((i+1)%epochs_sample) == 0:
        
        # Run N_MCMC_sims sampling simulations and take the mean acceptance fraction
        # and integrated autocorr. Not bothering with std/errors; seems unnecessary here.
        frac_accepted, integrated_autocorr = 0, 0
        for j in range(N_MCMC_sims):
            results = sample.sample(model)[:2]
            frac_accepted += results[0]
            integrated_autocorr += results[1]
        frac_accepted /= N_MCMC_sims
        integrated_autocorr /= N_MCMC_sims

        # Save results
        current_time = time.time() - start_time + restart_time
        out_array[i_out, :] = current_time, loss, frac_accepted, integrated_autocorr
        
        i_out += 1

# Save model
torch.save(model.state_dict(), model_dir+argv[1]+'.pt')

# Save convergence data
if initial_save:
    np.savetxt(training_data_dir+argv[1]+'.txt', out_array, header=header)    
else:
    print("appending..")
    with open(training_data_dir+argv[1]+'.txt', 'ab') as f:
        np.savetxt(f, out_array)    

 
