"""
geometry.py

Module containing transformations related to geometry.
"""
from math import ceil

import torch

class ShiftsMismatchError(Exception): pass

def get_shift(
        length: int,
        shifts: tuple = (1, 1),
        dims: tuple = (0, 1),
        ) -> torch.Tensor:
    r"""Given length, which refers to size of a 2D state (length * length)
    returns a 2x(length^2) tensor where each row gives the 2 nearest neighbours
    to a flattened state which has been split into (\phi_even, \phi_odd) where
    even/odd refer to parity of the site.

    Parameters
    ----------
    length: int
        Defines size of 2D state (length * length)
    shifts: tuple
        a tuple of shifts to be applied. Each element represents a shift and can
        either be an integer (if the shift is in a single dimension) or a tuple
        if the shift is applied simultaneously in multiple dimensions.
        By default it is set to have two shifts, left 1 and up 1 (which in turn
        allow for right and down nearest neighbours)
    dims: tuple
        a tuple of dimensions to apply `shifts` to. As with shift, each element
        in dim can itself be a tuple which indicates that multiple shifts will
        be applied in multiple dimensions simultaneously.
    Returns
    -------
    shift: torch.Tensor
        Tensor which can be used to index flattened, split states such that

            state = tensor([\phi_even, \phi_odd]),

        then state[shift] will return a 2xlength tensor:

            state[shift] -> tensor([[neighbour right], [neighbour down]])

    Example
    -------
    Consider the small example of 2x2 state:

    >>> state_2d = torch.arange(4).view(2, 2)
    >>> state_2d
    tensor([[0, 1],
            [2, 3]])

    even sites are [0, 3], odd sites are [1, 2]

    >>> state_split = torch.tensor([0, 3, 1, 2])
    >>> shift = get_shift(2)
    >>> state_split[shift]
    tensor([[1, 2, 0, 3],
            [2, 1, 3, 0]])

    correct nearest neighbours (left and down) are given in each row respectively

    to see how multiple shifts works, consider the shift (1, 1)
    (left one, up one)

    >>> shift = get_shift(2, shifts=((1, 1),), dims=((0, 1),))
    >>> state_split[shift]
    tensor([[3, 0, 2, 1]])

    we see that each element of shifts and dims can perform multiple shifts, in
    different dimensions at once.

    """
    if len(shifts) != len(dims):
        raise ShiftsMismatchError(
            "Number of shifts and number of dimensions: "
            f"{len(shifts)} and {len(shifts)} do not match."
        )
    # define a checkerboard
    checkerboard = torch.zeros((length, length), dtype=torch.int)
    # set even sites to 1
    checkerboard[1::2, 1::2] = 1
    checkerboard[::2, ::2] = 1

    # make 2d state-like matrix filled with corresponding indices in split-flat state
    splitind_like_state = torch.zeros((length, length), dtype=torch.int)
    splitind_like_state[checkerboard.bool()] = torch.arange(
        int(ceil(length**2/2)),
        dtype=torch.int,
    )
    splitind_like_state[~checkerboard.bool()] = torch.arange(
        int(ceil(length**2/2)),
        length**2,
        dtype=torch.int,
    )

    flat_checker = checkerboard.flatten()
    # make split-flat state like object with corresponding indices in flat state
    out_ind = torch.cat(
        [torch.where(flat_checker == 1)[0], torch.where(flat_checker == 0)[0]],
        dim=0,
    )

    shift_index = torch.zeros(len(shifts), length*length, dtype=torch.long)
    for i, (shift, dim) in enumerate(zip(shifts, dims)):
        # each shift, roll the 2d state-like indices and then flatten and split
        shift_index[i, :] = splitind_like_state.roll(shift, dims=dim).flatten()[out_ind]
    return shift_index
