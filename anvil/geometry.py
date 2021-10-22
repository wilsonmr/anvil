# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
geometry.py

Module containing transformations related to geometry.
"""
import itertools

import torch


class ShiftsMismatchError(Exception):
    pass


class Geometry2D:
    """Define the 2D geometry and the shifts in the two Cartesian directions."""

    def __init__(self, length):
        self.length = length
        self.volume = length ** 2

        self._cartesian_grid = torch.arange(self.volume).view(self.length, self.length)

        checkerboard = torch.zeros((self.length, self.length)).bool()
        checkerboard[1::2, 1::2] = True
        checkerboard[::2, ::2] = True
        self._checkerboard = checkerboard

    @property
    def checkerboard(self) -> torch.BoolTensor:
        """Returns 2d mask that selects the 'even' sites of the lattice."""
        return self._checkerboard

    def get_shift(self, shifts: tuple = (1, 1), dims: tuple = (0, 1)) -> torch.Tensor:
        r"""Provided with a set of `shifts` and `dims` which are tuples of equal
        length N, the number of shifts. For the given system length used to
        instance Geometry2D, which refers to size of a 2D state (length * length)
        returns a tensor, size (N, length^2). Row i of the returned tensor indexes a
        flattened state which has been shifted by shift[i] in dimension dims[i].
        The shifts are performed on the 2D cartesian states.

        Element i of shifts and dims can either both be an integer or both be a
        tuple of equal length. In the case that the element is a tuple, it
        represents multiple simultaneous shifts in multiple dimensions.

        By default shifts = (1, 1) and dims = (0, 1) and the resulting tensor indexes
        the nearest neighbours above and to the left respectively. This convention
        is according to to torch.roll

        Parameters
        ----------
        shifts
            a tuple of shifts to be applied. Each element represents a shift and can
            either be an integer (if the shift is in a single dimension) or a tuple
            if the shift is applied simultaneously in multiple dimensions (see
            Examples).

        dims
            a tuple of dimensions to apply `shifts` to. As with shift, each element
            in dim can itself be a tuple which indicates that multiple shifts will
            be applied in multiple dimensions simultaneously. Note that
            corresponding entries of dims and shifts must also match (either both
            ints or both tuples of same length).

        Returns
        -------
        torch.Tensor
            Tensor which can be used to index flattened states

        Example
        -------
        Consider the small example of 2x2 state:

        >>> state_2d = torch.arange(4).view(2, 2)
        >>> state_2d
        tensor([[0, 1],
                [2, 3]])

        If we flatten the state and index using the default shift, the output is a
        tensor for which the i'th column contains the indices of the nearest
        neighbours above and to the left of the i'th lattice sites.

        >>> phi = state_2d.flatten()
        >>> geom = Geometry2D(2)
        >>> shift = geom.get_shift()
        >>> phi[shift]
        tensor([[2, 3, 0, 1],
                [1, 0, 3, 2]])

        to see how multiple shifts works, consider the shift (1, 1) in dimensions
        (0, 1): up one, left one

        >>> shift = geom.get_shift(shifts=((1, 1),), dims=((0, 1),))
        >>> phi[shift]
        tensor([[3, 2, 1, 0]])

        Notes
        -----
        The conventions for how the shifts are applied are according the torch.roll
        function, shift = +ve rolls the state in a direction that corresponds
        to ascending index when using standard python indexing.

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
            # each shift, roll the 2d state-like indices and then flatten
            shift_index[i, :] = self._cartesian_grid.roll(shift, dims=dim).flatten()
        return shift_index

    def two_point_iterator(self):
        """Generator which yields all the lattice shifts as one-dimensional tensors.

        Yields
        ------
        torch.Tensor
            one-dimensional tensor containing the shift indices for each lattice shift.

        Notes
        -----
        The order in which the shifts are generated is defined by the lexicographical
        order of the Cartesian product of one-dimensional shifts. See :py:func:`itertools.product`.
        """
        for shift_cart in itertools.product(range(self.length), range(self.length)):
            yield self.get_shift((shift_cart,), ((0, 1),)).flatten()
