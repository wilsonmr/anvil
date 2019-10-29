"""
grid.py

Module containing functions related to the grid, and transformations between the original
2-d grid and the 1-d chequerboard "split state" as required by the neural networks.

The only module not to import params.py - currently it's imported *by* params.py for
the phi4 action computation.
"""
import torch

def make_checkerboard(L: int):
    checkerboard = torch.zeros((L,L), dtype=bool)
    # set even sites to 1
    checkerboard[1::2, 1::2] = True
    checkerboard[ ::2,  ::2] = True
    
    return checkerboard

def get_splitind_like_state(L: int):
    """An LxL array of indices corresponding to the flat-split state."""
    
    checkerboard = make_checkerboard(L)

    # make 2d state-like matrix filled with corresponding indices in split-flat state
    indices = torch.zeros((L,L), dtype=torch.long)
    indices[checkerboard] = torch.arange(0, int(L**2/2), dtype=torch.long)
    indices[~checkerboard] = torch.arange(int(L**2/2), L**2, dtype=torch.long)
    
    return indices


def get_2d_ind_wrapped(index, L):
    """Given any pair of indices (i,j), returns a pair of indices within
    the bounds of the periodic L*L lattice.
    
    Can be easily generalised to higher dimensions.
    """
    
    index_wrapped = (index[0]%L, index[1]%L)
    return index_wrapped


def get_stateind_like_split(L: int):
    """A 1xL^2 "split" array of tuples corresponding to the state indices."""

    indices_even = [ (i//L, i%L+(i//L)%2) for i in range(0,L**2,2) ]
    indices_odd  = [ (i//L, i%L-(i//L)%2) for i in range(1,L**2,2) ]

    return indices_even + indices_odd

def get_flatind_like_split(L: int):
    """Returns 1xL^2 tensor of indices corresponding to the flattened 2d state,
    in split-flat format."""
    
    checkerboard = make_checkerboard(L)

    # Make split-flat state-like object with corresponding indices in flat state
    flat_checker = checkerboard.flatten()
    indices = torch.cat(
        [torch.where(flat_checker==1)[0], torch.where(flat_checker==0)[0]],
        dim=0,
    )

    return indices

def neighbours(L: int):
    """Returns a 4xL^2 tensor where each row contains the 4 nearest neighbours, in
    the flat-split format."""
        
    # LxL matrix of split-flat indices
    splitind_like_state = get_splitind_like_state(L)

    # 1xL split-flat tensor of flat indices - used for indexing flattened 2d state
    out_ind = get_flatind_like_split(L)

    direction_dimension = [
            (-1, 1),
            (1, 1),
            (-1,0),
            (1,0)
    ]
    neighbours = torch.zeros(4, L*L, dtype=torch.long)
    for i, (direction, dim) in enumerate(direction_dimension):
        # each shift, roll the 2d state-like indices and then flatten and split
        neighbours[i,:] = splitind_like_state.roll(direction, dims=dim).flatten()[out_ind]

    return neighbours





