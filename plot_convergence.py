"""
plot_convergence.py

Load data from single or multiple training runs to compare convergence, i.e.
loss, acceptance fraction and integrated autocorrelation time.
"""

import numpy as np
import matplotlib.pyplot as plt
from sys import argv

from params import *

# Which files are we loading?
file_list = ('L6na8.txt','L6na10.txt','L6na12.txt','L6na14.txt','L6na16.txt',
            'L4na18.txt','L4na20.txt','L4na22.txt','L4na24.txt',) # NOTE: needs to be a tuple even only one file, eg ('<filename>',)

# First line of file is a list of these simulation parameters
col_dict = {
        'L': 0,
        'N_BATCH': 1,
        'n_affine': 2,
        'affine_hidden_shape': 3,
        'epochs_sample': 4,
        'target_length': 5,
        'n_large': 6
        }
# Which parameter(s) vary between input files?
params_to_compare = ('n_affine',) # NOTE: needs to be a tuple even only one file, eg ('<param>',)

# Set up plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8,10))
ax3.set_xlabel('time (minutes)')
ax1.set_ylabel('loss')
ax2.set_ylabel('acceptance')
ax3.set_ylabel('int. autocorr')

# Load and plot data
for afile in file_list:
    # Load parameters from header
    with open(training_data_dir+afile, 'r') as f: aparams = f.readline().strip().split()[1:]
    
    # Load data as array
    data = np.loadtxt(training_data_dir+afile)
    time = data[:,0] / 60. # convert to minutes
    loss = data[:,1]
    facc = data[:,2]
    auto = data[:,3]
    
    # Make label
    label = ""
    for param in params_to_compare:
        label = label + param + "=" + aparams[col_dict[param]]
    
    ax1.plot(time, loss, 'o-', label=label)
    ax2.plot(time, facc, 'o-')
    ax3.plot(time, auto, 'o-')

ax1.legend(loc=1)
plt.tight_layout()
plt.show()
