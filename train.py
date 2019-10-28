"""
train.py

Run python3 train.py <model_name>
"""
import sys
import os
import time
from math import ceil, exp

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim

from norm_flow_pytorch import NormalisingFlow, shifted_kl
from params import *


# set seed, hopefully result is reproducible
torch.manual_seed(0)
   
model = NormalisingFlow(
    size_in=D, n_affine=n_affine,
    affine_hidden_shape=affine_hidden_shape)
    

# ---------- #
#  Training  #
# ---------- #
# create your optimizer and a scheduler
optimizer = optim.Adadelta(model.parameters(), lr=1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500)
# let's use tqdm to see progress
pbar = tqdm(range(epochs), desc=f"loss: N/A")
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
        pbar.set_description(f"loss: {loss.item()}")

    
if not os.path.exists('models'): os.makedirs('models')
torch.save(model.state_dict(), 'models/'+sys.argv[1])
 
