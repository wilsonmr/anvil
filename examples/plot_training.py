"""
plot_training.py
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
from sys import argv, exit
from os.path import exists

# --- Gloal parameters to be set by user --- #
# Choose x data from 'epochs' or 'minutes'
xdata_type = "epochs"
# Choose y data from 'loss', 'acceptance', 'tauint'
ydata_type = "acceptance"
# Choose y error bars from 'std_acceptance', 'std_tauint', None
yerr_type = None
# Which parameters are we comparing?
params_to_compare = ("hidden_nodes", "n_batch")
# Name of output directory from anvil-train (don't include '/' !)
train_dir = "training_output"

col_dict = {  # corresponds with columns of training_data.out, set in train_to_acceptance.sh
    "epochs": 0,
    "minutes": 1,
    "loss": 2,
    "acceptance": 3,
    "std_acceptance": 4,
    "tauint": 5,
    "std_tauint": 6,
}

# --- Specify data directories here or as command line args --- #
data_dirs = [
    "dir1",
    "dir2"
]
if len(argv) > 1:  # Overwrite if command line args
    data_dirs = argv[1:]

# --- Bad input handling --- #
def nae_exists(target):
    return f"Error: {target} does not exist!"

def bad_input(data_dir):
    if not exists(data_dir):
        print(nae_exists(data_dir))
        return 1
    elif not exists(data_dir+train_dir):
        print(nae_exists(data_dir+train_dir))
        return 1
    else:
        tally = 0
        if not exists(data_dir+train_dir+"/training_data.out"):
            print(nae_exists(data_dir+train_dir+"/training_data.out"))
            tally += 1
        elif not exists(data_dir+train_dir+"/runcard.yml"):
            print(nae_exists(data_dir+train_dir+"/runcard.yml"))
            tally += 1
        return tally

bad_input_tally = 0
for data_dir in data_dirs:
     bad_input_tally += bad_input(data_dir)
if bad_input_tally > 0:
    print("Exiting...")
    exit(1)

# --- Function definitions --- #
def get_data(data_dir):
    """Get x and y data"""
    data = np.loadtxt(data_dir + train_dir + "/training_data.out")
    xdata = data[:, col_dict[xdata_type]]
    ydata = data[:, col_dict[ydata_type]]
    if yerr_type == None:
        return xdata, ydata
    else:
        yerr = data[col_dict[yerr_type]]
        return xdata, ydata, yerr


def get_label(data_dir):
    """Get info from runcard to create label for plot"""
    try:
        with open(data_dir + train_dir + "/runcard.yml", "r") as stream:
            runcard = yaml.safe_load(stream)
    except FileNotFoundError:
        print("Error: "+data_dir+train_dir+"/runcard.yml does not exist.")
        print("Exiting..."); exit(1)
    label = ""
    for param in params_to_compare:
        label += param + ":" + str(runcard[param]) + ", "
    label = label[:-2]
    return label


# --- Create plot --- #
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title("Training data")

if ydata_type == "tauint":
    ydata_label = r"\tau_{int}"
else:
    ydata_label = ydata_type
ax.set_xlabel(xdata_type)
ax.set_ylabel(ydata_label)

for data_dir in data_dirs:
    if isdir(data_dir) == False:
        print(f"Error: no such directory {data_dir}"); exit(1)
    if data_dir[-1] != '/': data_dir += '/'
    label = get_label(data_dir)
    if yerr_type == None:
        xdata, ydata = get_data(data_dir)
        ax.plot(xdata, ydata, label=label)
    else:
        xdata, ydata, yerr = get_data(data_dir)
        ax.errorbar(xdata, ydata, yerr=yerr, label=label)

ax.legend()
fig.tight_layout()
plt.show()
