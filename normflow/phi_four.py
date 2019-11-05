"""
phi_four.py

script which can train and load models for phi^4 action
"""
import sys
import time

import torch

from normflow.models import NormalisingFlow
from normflow.observables import PhiFourAction, print_plot_observables
from normflow.train import train
from normflow.sample import sample

class InvalidArgsError(Exception):
    pass

def main():
    """main loop for phi_four.py"""
    length = 6
    n_units = length ** 2
    m_sq, lam = -4, 6.975
    # set seed, hopefully result is reproducible
    # torch.manual_seed(0)
    action = PhiFourAction(length, m_sq, lam)
    # define simple mode, each network is single layered
    if not ((len(sys.argv) == 3) and (sys.argv[1] in ["train", "load"])):
        raise InvalidArgsError(
            "Pass `train` and a model name to train new model or "
            "`load` and model name to load existing model"
        )

    if sys.argv[1] == "train":
        model = NormalisingFlow(size_in=n_units, n_affine=8, affine_hidden_shape=(32,))
        epochs = 10000  # Gives a decent enough approx.
        train(model, action, epochs)
        torch.save(model.state_dict(), "models/" + sys.argv[2])
    elif sys.argv[1] == "load":
        model = NormalisingFlow(size_in=n_units, n_affine=8, affine_hidden_shape=(32,))
        model.load_state_dict(torch.load("models/" + sys.argv[2]))
    target_length = 100000  # Number of samples we want

    start_time = time.time()
    # Perform Metroplis-Hastings sampling
    sample_dist = sample(model, action, target_length)
    print_plot_observables(sample_dist)
    print(
        f"Time to run MC for a chain of {target_length} "
        f"samples on an L={length} lattice: {time.time() - start_time} seconds"
    )


if __name__ == "__main__":
    main()
