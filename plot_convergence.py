"""
plot_convergence.py

Load data from single or multiple training runs to compare convergence, i.e.
loss, acceptance fraction and integrated autocorrelation time.
"""

import numpy as np
import matplotlib.pyplot as plt
from sys import argv
import ast

from params import *

# Which files are we loading?
# Files to be found in 'training_data_dir', specified in params.py
file_list = ('L4B1000.txt',)

# Which parameter(s) vary between input files? (will be shown on plot legend)
params_to_compare = ('N_BATCH',)

# Set up plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8,10))
ax3.set_xlabel('time (minutes)')
ax1.set_ylabel('loss')
ax2.set_ylabel('acceptance')
ax3.set_ylabel('int. autocorr')

# Load and plot data
for afile in file_list:
    # Load parameters from header
    with open(training_data_dir+afile, 'r') as f:
        # strip "# " from front then evaluate as dict
        aparams = ast.literal_eval( f.readline().strip()[2:] )
    
    # Load data as array
    data = np.loadtxt(training_data_dir+afile)
    time = data[:,0] / 60. # convert to minutes
    loss = data[:,1]
    facc = data[:,2]
    auto = data[:,3]
    
    # Make label
    label = ""
    for param in params_to_compare:
        label = label + param + "=" + str(aparams[param])
     
    ax1.plot(time, loss, 'o-', label=label)
    ax2.plot(time, facc, 'o-')
    ax3.plot(time, auto, 'o-')

ax1.legend(loc=1)
plt.tight_layout()
plt.show()
