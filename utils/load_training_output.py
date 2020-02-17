"""
load_data.py
"""
import numpy as np
import yaml
from sys import exit
from os.path import exists
import filecmp


col_dict = {  # corresponds to columns of training_data.out, set in train_to_acceptance.sh
    "epochs": 0,
    "time": 1,
    "loss": 2,
    "acceptance": 3,
    "std_acceptance": 4,
    "tauint": 5,
    "std_tauint": 6,
}
lr_col_dict = {  # corresponds to columns of learning_rate.out, set in train_to_acceptance.sh
    "epochs": 0,
    "time": 1,
    "lr": 2,
}
time_dict = {"seconds": 1, "minutes": 60, "hours": 3600}


def bad_input(target):
    """Return True/False depending on whether target exists."""
    if not exists(target):
        print(f"Warning: {target} does not exist!")
        return True
    else:
        return False


def check_all_inputs(data_dirs, required_files):
    """Get a tally of how many input directories would lead to FileNotFoundError."""
    bad_input_tally = 0
    for data_dir in data_dirs:
        bad_dir = bad_input(data_dir)
        bad_input_tally += int(bad_dir)
        if not bad_dir:
            for f in required_files:
                bad_input_tally += int(bad_input(data_dir + f))
    if bad_input_tally > 0:
        print("Exiting...")
        exit(1)


# Class for training output from a given directory
class TrainingOutput:
    def __init__(self, data_dir):
        self.check_input(data_dir)
        self.data_dir = data_dir
        self.data_file = self.data_dir + "training_data.out"
        self.runcard = self.data_dir + "runcard.yml"
        self.lr_file = self.data_dir + "learning_rate.out"

        self.check_input(self.data_file)
        self.check_input(self.runcard)

        self.load_runcard()
        # Don't want to load output data on construction
        # in case it's big
        self.loaded_data = None
        self.loaded_lr_data = None

    def check_input(self, target, fatal=True):
        """Exit gracefully and informatively on fatal FileNotFoundError."""
        if bad_input(target) == True:
            if fatal == True:
                print("Fatal error. Exiting...")
                exit(1)
            else:
                print("Error not fatal. Continuing...")
        return

    def load_data(self):
        self.loaded_data = np.loadtxt(self.data_file)
        return

    def load_runcard(self):
        with open(self.runcard, "r") as stream:
            self.loaded_runcard = yaml.safe_load(stream)
        return

    def load_lr_data(self):
        self.check_input(self.lr_file)
        self.loaded_lr_data = np.loadtxt(self.lr_file)
        return

    def get_data(self, data_type, tunits="hours"):
        """Return a single column of data from the training output."""
        if type(self.loaded_data) == type(None):
            self.load_data()
        col = self.loaded_data[:, col_dict[data_type]]
        if data_type == "time":
            col = col / time_dict[tunits]  # time in sec/min/hour
        return col

    def get_lr_data(self, data_type, tunits="hours"):
        """Return a single column of data from the learning rate output."""
        if type(self.loaded_lr_data) == type(None):
            self.load_lr_data()
        col = self.loaded_lr_data[:, lr_col_dict[data_type]]
        if data_type == "time":
            col = col / time_dict[tunits]
        return col

    def get_label(self, params):
        """Get info from runcard to create label for plot."""
        if params == None:
            return self.data_dir.strip("/")
        label = ""
        for param in params:
            label += param + ":" + str(self.loaded_runcard[param]) + ", "
        label = label[:-2]
        return label

    def _flatten_runcard_dict(self, d):
        """Turn dict of dicts into flattened dict."""

        def expand(k, v):
            if isinstance(v, dict):
                return [
                    (k + "." + kk, vv)
                    for kk, vv in self._flatten_runcard_dict(v).items()
                ]
            else:
                return [(k, v)]

        items = [item for k, v in d.items() for item in expand(k, v)]
        return dict(items)

    def get_runcard_params(self):
        """Return flattened runcard dict."""
        return self._flatten_runcard_dict(self.loaded_runcard)


def organise_repeats(training_outputs):
    """Return list of lists, where each sublist contains TrainingOutput objects
    which have the same runcard"""
    if len(training_outputs) < 2:
        return training_outputs.strip("/")

    groups = [
        [training_outputs[0],],
    ]
    for training_output in training_outputs[1:]:
        for group in groups:
            if filecmp.cmp(training_output.runcard, group[0].runcard) is True:
                group.append(training_output)
                print("same!")
                break
        groups.append(
            [training_output,]
        )
    return groups
