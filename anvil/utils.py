# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
import numpy as np
import multiprocessing as mp
from itertools import islice
from functools import wraps
from math import ceil
import torch
import sys
import torch


class Multiprocessing:
    """Class which implements multiprocessing of a function given a number
    inputs for that function.

    Parameters
    ----------
    func
        the function to be executed multiple times
    generator
        something which, when called, returns a generator object that contains
        the parameters for the function.

    Notes
    -----
    Does not rely on multiprocessing.Pool since that does not work with
    instance methods without considerable extra effort (it cannot pickle them).
    This means that the multiprocessing is not supported on Mac.

    """

    def __init__(self, func, generator, use_multiprocessing: bool):
        self.func = func
        self.generator = generator

        self.n_iters = sum(1 for _ in generator())

        if not use_multiprocessing:
            self.n_cores = 1
        else:
            self.n_cores = mp.cpu_count()

        self.max_chunk = ceil(self.n_iters / self.n_cores)

    def target(self, k: int, output_dict: dict) -> None:
        """Function to be executed for each process."""
        generator_k = islice(
            self.generator(),
            k * self.max_chunk,
            min((k + 1) * self.max_chunk, self.n_iters),
        )
        i_glob = k * self.max_chunk  # global index
        for i, args in enumerate(generator_k):
            output_dict[i_glob + i] = self.func(args)

    def __call__(self) -> dict:
        """Returns a dictionary containing the function outputs for each
        set of parameters taken from the generator. The dictionary keys are
        integers which label the order of parameter sets in the generator."""
        # don't use mp if single core.
        if self.n_cores == 1:
            output_dict = dict()
            self.target(0, output_dict)
        else:
            manager = mp.Manager()
            output_dict = manager.dict()

            procs = []
            for k in range(self.n_cores):
                p = mp.Process(
                    target=self.target,
                    args=(
                        k,
                        output_dict,
                    ),
                )
                procs.append(p)
                p.start()

            # Kill the zombies
            for p in procs:
                p.join()

        return output_dict


def bootstrap_sample(data: np.ndarray, bootstrap_sample_size: int, seed=None) -> np.ndarray:
    """Resample a provided array to generate a bootstrap sample.

    The last dimension of the array will be one that is bootstrapped, and each
    member of the bootstrap sample will have the same shape: ``data.shape`` .

    The boostrap dimension will be inserted at position ``[-2]`` in the output
    array.

    Parameters
    ----------
    data
        Array containing the data to be resampled.
    bootstrap_sample_size
        Size of the bootstrap sample, i.e. number of times to resample the data.
    seed
        Optional seed for the rng which generates the bootstrap indices, for
        reproducibility purposes and to allow different terms in a single
        expression to be passed to this function independently.

    Returns
    -------
    np.ndarray
        Array containing the bootstrap sample, dimensions
        ``(*data.shape[:-1], bootstrap_sample_size, data.shape[-1])`` .
    """
    rng = np.random.default_rng(seed=seed)
    *dims, data_size = data.shape

    sample = []
    for j in range(bootstrap_sample_size):
        boot_index = rng.integers(low=0, high=data_size, size=data_size)
        sample.append(data[..., boot_index])

    return np.stack(sample, axis=-2)


def get_num_parameters(model) -> int:
    """Returns the number of trainable parameters in a model.

    Reference: github.com/bayesiains/nflows
    """
    num = 0
    for parameter in model.parameters():
        num += torch.numel(parameter)
    return num


def handler(signum, frame) -> None:
    """Handles keyboard interruptions and terminations and exits in such a way that,
    if the program is currently inside a try-except-finally block, the finally clause
    will be executed."""
    sys.exit(1)
