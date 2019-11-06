"""
phi_four.py

script which can train and load models for phi^4 action
"""
import sys
import time
import os

import torch

from normflow.config import ConfigParser
from normflow.train import train
from normflow.sample import sample
from normflow.observables import Observables


class InvalidArgsError(Exception):
    pass


def main():
    """main loop for phi_four.py"""
    config = ConfigParser(sys.argv[1])
    outpath = sys.argv[2]
    os.mkdir(outpath)

    model, action, geometry, train_spec, sample_spec = config.resolve()
    if train_spec is not None:
        train(model, action, **train_spec, outpath=outpath)
        torch.save(model.state_dict(), f"{outpath}/final.pt")
    if sample_spec is not None:
        start_time = time.time()
        # Perform Metroplis-Hastings sampling
        sample_dist = sample(model, action, sample_spec['chain_length'])
        print(
            f"Time to run MC for a chain of {sample_spec['chain_length']} "
            f"samples on an L={geometry.length} lattice: {time.time() - start_time} seconds"
        )
        obs_class = Observables(sample_dist, geometry, outpath)
        for obs in sample_spec['observables']:
            _ = getattr(obs_class, obs)()


if __name__ == "__main__":
    main()
