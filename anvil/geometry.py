"""
geometry.py

Module containing transformations related to geometry.
"""
from math import ceil
from itertools import product

import torch


class ShiftsMismatchError(Exception):
    pass


class Geometry2D:
    """Define the 2D geometry and the shifts in the two Cartesian
    directions. The fields are stored in a one-dimensional array of size
    length*length, assuming that the first Na entries correspond to the sites
    that are updated by the affine transformation, and the remaining Nb
    entries correspond to the sites that are left unchanged.

    phi = |... phiA ...|... phiB ...|

    using the notation in https://arxiv.org/pdf/2003.06413.pdf We call this
    representation of the field a 'split' representation.

    """

    def __init__(self, length):
        self.length = length
        self.volume = length ** 2  # NOTE: currently alias of lattice_size, which is not ideal
        # TODO: Make the split pattern flexible and controllable at level of instance
        checkerboard = torch.zeros((self.length, self.length), dtype=bool)
        checkerboard[1::2, 1::2] = True
        checkerboard[::2, ::2] = True
        self.checkerboard = checkerboard
        self.flat_checker = self.checkerboard.flatten()
        # make split-flat state like object with corresponding indices in flat state
        self.splitcart = self._splitcart()
        self.lexisplit = self._lexisplit()

    def _splitcart(self):
        """Split to Cartesian. Internal function which returns a 2-D grid of
        integers of size length*length. The element (x,y) of the grid contains
        the index that identifies the position of the site (x,y) in the split
        representation.

        """
        splitcart = torch.zeros((self.length, self.length), dtype=torch.long)
        n_a = self.checkerboard.sum().item()
        splitcart[self.checkerboard] = torch.arange(
            n_a, dtype=torch.long
        )
        splitcart[~self.checkerboard] = torch.arange(
            n_a, self.length ** 2, dtype=torch.long
        )
        return splitcart

    def _lexisplit(self):
        """Lexicographic to Split. Internal function that returns a 1-D tensor of
        integers of size length*length. The element i of the tensor contains the
        index that identifies the position of the site i in the split
        representation, where i is the lexicographic index of the site.

        The element i of the tensor contains the lexicographic index of the
        site i in the split representation. Where the lexicographic index refers
        to the original 2-D state.

        Consider for example the orignal 2-D state where each element is the
        lexicographic index of that site.

            phi_2D = [[0, 1],
                      [2, 3]]

        which is then split according to checkerboard pattern

            phi_split = [phiA, phiB]
                      = [0, 3, 1, 2]

        Since each site's value is its lexicographic index, then we can write
        down immediately what lexisplit will return.

            lexisplit = [0, 3, 1, 2]

        """
        lexisplit = torch.cat(
            [torch.where(self.flat_checker)[0], torch.where(~self.flat_checker)[0]],
            dim=0,
        )
        return lexisplit

    def get_shift(self, shifts: tuple = (1, 1), dims: tuple = (0, 1)) -> torch.Tensor:
        r"""Provided with a set of `shifts` and `dims` which are tuples of equal
        length N, the number of shifts. For the given system length used to
        instance Geometry2D, which refers to size of a 2D state (length * length)
        returns a tensor, size (N, length^2). Row i of the returned tensor indexes a
        split state \phi = (\phiA, \phiB) which has been shifted by shift[i] in
        dimension dims[i]. The shifts are performed on the 2D cartesian states.

        element i of shifts and dims can either both be an integer or both be a
        tuple of equal length. In the case that the element is a tuple, it
        represents multiple simultaneous shifts in multiple dimensions.

        By default shifts = (1, 1) and dims = (0, 1) and the resulting tensor indexes
        the nearest neighbours above and to the left respectively. This convention
        is according to to torch.roll

        Parameters
        ----------
        shifts: tuple
            a tuple of shifts to be applied. Each element represents a shift and can
            either be an integer (if the shift is in a single dimension) or a tuple
            if the shift is applied simultaneously in multiple dimensions (see
            Examples).

        dims: tuple
            a tuple of dimensions to apply `shifts` to. As with shift, each element
            in dim can itself be a tuple which indicates that multiple shifts will
            be applied in multiple dimensions simultaneously. Note that
            corresponding entries of dims and shifts must also match (either both
            ints or both tuples of same length).

        Returns
        -------
        shift: torch.Tensor
            Tensor which can be used to index split states such that

                state = tensor([\phiA, \phiB]),

            then state[shift] will return a 2xlength tensor:

                state[shift] -> tensor([[neighbour right], [neighbour down]])

        Example
        -------
        Consider the small example of 2x2 state:

        >>> state_2d = torch.arange(4).view(2, 2)
        >>> state_2d
        tensor([[0, 1],
                [2, 3]])

        If we use a checkerboard pattern to split state into \phiA and \phiB
        then \phiA = [0, 3] and \phiB = [1, 2]

        >>> phi = torch.tensor([0, 3, 1, 2])
        >>> geom = Geometry2D(2)
        >>> shift = geom.get_shift()
        >>> phi[shift]
        tensor([[2, 1, 3, 0],
                [1, 2, 0, 3]])

        to see how multiple shifts works, consider the shift (1, 1) in dimensions
        (0, 1): up one, left one

        >>> shift = geom.get_shift(shifts=((1, 1),), dims=((0, 1),))
        >>> phi[shift]
        tensor([[3, 0, 2, 1]])

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
            # each shift, roll the 2d state-like indices and then flatten and split
            shift_index[i, :] = self.splitcart.roll(shift, dims=dim).flatten()[
                self.lexisplit
            ]
        return shift_index

    def two_point_iterator(self):
        """Generator which yields all the lattice shifts as one-dimensional tensors.

        Notes
        -----
        The order in which the shifts are generated is defined by the lexicographical
        order of the Cartesian product of one-dimensional shifts. See the documentation
        for itertools.product for details.
        """
        for shift_cart in product(range(self.length), range(self.length)):
            yield self.get_shift((shift_cart,), ((0, 1),)).flatten()

