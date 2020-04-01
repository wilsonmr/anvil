"""
geometry.py

Module containing transformations related to geometry.
"""
from math import ceil

import torch


class ShiftsMismatchError(Exception):
    pass


class Geometry2D:
    """Define the 2D geometry, which can return shifts which index flattened
    returning shifted flattened states, where the shift is some shift on the
    original lattice
    """

    def __init__(self, length):
        self.length = length
        # Hard code the checkerboard pattern, could be passed to class.
        checkerboard = torch.zeros((self.length, self.length), dtype=bool)
        checkerboard[1::2, 1::2] = True
        checkerboard[::2, ::2] = True
        self.checkerboard = checkerboard
        self.flat_checker = self.checkerboard.flatten()
        # make split-flat state like object with corresponding indices in flat state
        self.flat_ind_like_split = torch.cat(
            [
                torch.where(checkerboard.flatten())[0],
                torch.where(~checkerboard.flatten())[0],
            ],
            dim=0,
        )
        self.split_ind_like_state = self._split_indices_like_state()

    def _split_indices_like_state(self):
        """Internal function for calculating the split index like state, which
        is the index of phi_a and phi_b in the flattened state, arranged in the
        shape of the 2D state, used in get_shift
        """
        splitind_like_state = torch.zeros((self.length, self.length), dtype=torch.int)
        splitind_like_state[self.checkerboard] = torch.arange(
            int(ceil(self.length ** 2 / 2)), dtype=torch.int
        )
        splitind_like_state[~self.checkerboard] = torch.arange(
            int(ceil(self.length ** 2 / 2)), self.length ** 2, dtype=torch.int
        )
        return splitind_like_state

    def get_shift(self, shifts: tuple = (1, 1), dims: tuple = (0, 1)) -> torch.Tensor:
        r"""Given length, which refers to size of a 2D state (length * length)
        returns a Nx(length^2) tensor where N is the length of `shifts` and `dims`
        (which must be equal). Each row of the returned tensor indexes a flattened
        split state \phi = (\phi_even, \phi_odd) which is split according to a
        checkerboard geometry (even and odd refer to parity of the site). The
        indices refer to shifts on the states in their original 2D form. By default
        N = 2 and get_shift simply returns the right and down nearest neighbours.

        Parameters
        ----------
        shifts: tuple
            a tuple of shifts to be applied. Each element represents a shift and can
            either be an integer (if the shift is in a single dimension) or a tuple
            if the shift is applied simultaneously in multiple dimensions (see
            Examples). By default it is set to have two shifts which give right and
            down nearest neighbours
        dims: tuple
            a tuple of dimensions to apply `shifts` to. As with shift, each element
            in dim can itself be a tuple which indicates that multiple shifts will
            be applied in multiple dimensions simultaneously. Note that
            corresponding entries of dims and shifts must also match (either both
            ints or both tuples of same length).
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

        correct nearest neighbours in reference to the original `state_2d` (left and
        down) are given in each row respectively

        to see how multiple shifts works, consider the shift (1, 1)
        (left one, up one)

        >>> shift = get_shift(2, shifts=((1, 1),), dims=((0, 1),))
        >>> state_split[shift]
        tensor([[3, 0, 2, 1]])

        we see that each element of shifts and dims can perform multiple shifts, in
        different dimensions at once.

        Notes
        -----
        The conventions for how the shifts are applied are according the torch.roll
        function, shift = +ve rolls the state left and so the indices will refer to
        lattice sights to the right.

        See Also
        --------
        torch.roll: https://pytorch.org/docs/stable/torch.html#torch.roll

        """
        if len(shifts) != len(dims):
            raise ShiftsMismatchError(
                "Number of shifts and number of dimensions: "
                f"{len(shifts)} and {len(dims)} do not match."
            )

        shift_index = torch.zeros(
            len(shifts), self.length * self.length, dtype=torch.long
        )
        for i, (shift, dim) in enumerate(zip(shifts, dims)):
            # each shift, roll the 2d state-like indices and then flatten and split
            shift_index[i, :] = self.split_ind_like_state.roll(
                shift, dims=dim
            ).flatten()[self.flat_ind_like_split]
        return shift_index
