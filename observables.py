"""
observables.py

"""
from math import sqrt
import numpy as np
import torch
from itertools import product

import grid
from params import *

data = torch.load("sample_dist.pt")


def two_point_func(phi: torch.Tensor, x: tuple) -> float:
    """Takes a stack of split states \phi = (\phi_a, \phi_b), shape (N_states, D),
    and computes the (split) two-point function for a given 2d separation x,
    by averaging over the stack.

    See Eq. (22)
    """
    
    G = 0
    for p in range(D):
        # First, get the 2d coordinates of this point
        y = grid.get_stateind_like_split(L)[p]

        # Next, get flat-split indices of the points 'x' away in the 2D state
        yplusx1 = grid.get_2d_ind_wrapped( (y[0]+x[0], y[1]+x[1]), L )
        yplusx2 = grid.get_2d_ind_wrapped( (y[0]-x[0], y[1]+x[1]), L )
        yplusx3 = grid.get_2d_ind_wrapped( (y[0]+x[0], y[1]-x[1]), L )
        yplusx4 = grid.get_2d_ind_wrapped( (y[0]-x[0], y[1]-x[1]), L )
        
        # Convert back to 1d split indices
        q1 = grid.get_splitind_like_state(L)[yplusx1[0], yplusx1[1]]
        q2 = grid.get_splitind_like_state(L)[yplusx2[0], yplusx2[1]]
        q3 = grid.get_splitind_like_state(L)[yplusx3[0], yplusx3[1]]
        q4 = grid.get_splitind_like_state(L)[yplusx4[0], yplusx4[1]]
        q = [q1,q2,q3,q4]

        # Finally, Eq. (22) average over states
        G += (0.25*(
                torch.mul(phi[:,p], phi[:,q1]).mean() +
                torch.mul(phi[:,p], phi[:,q2]).mean() +
                torch.mul(phi[:,p], phi[:,q3]).mean() +
                torch.mul(phi[:,p], phi[:,q4]).mean() )     # <phi(y)phi(x+y)>
              - phi[:,p].mean() * phi[:,q].mean()           # <phi(y)><phi(x+y)>
            )
    
    G = G / D   # normalise (i.e. divide by number of points we've summed over)
    return float(G)


def zero_mom_2pf(phi: torch.Tensor) -> torch.Tensor:
    """Fourier transform of the two-point function at zero momentum,
    which (as far as I can work out) is just the sum along one axis."""
    
    #########################################################################
    #### PROBLEM! G_tilde_0 is not coming out as convex. Possibly due to ####
    ###    insufficient training, or very possibly a mistake in my code. ####
    #########################################################################
    G_tilde_0 = torch.zeros(L)
    for t in range(L):
        for x in range(L):
            G_tilde_0[t] += two_point_func(phi,(x,t))

    
    return G_tilde_0


#zm2pf = zero_mom_2pf(data)
#print(zm2pf)
#import matplotlib.pyplot as plt

#plt.plot(zm2pf)
#plt.show()

def susceptibility(phi: torch.Tensor):
    """See Eq. (25).""" 
    chi = 0
    # Sum over all possible x from (0,0) up to (3,3)
    for x in list( product(range(L), repeat=2) ): 
        chi += two_point_func(phi, x)
    
    return chi

def ising_energy_density(phi: torch.Tensor):
    """See Eq. (26)."""
    # Only consider x=(0,1) and x=(1,0)
    E = ( two_point_func(phi, (0,1)) + two_point_func(phi, (1,0)) ) / 2.

    return E


def pole_mass(phi: torch.Tensor):
    """An estimator for the pole mass.

    See Eq. (28)
    """

    G_tilde_0 = zero_mom_2pf(phi).numpy()
    

    # NaN's if G_tilde_0 is not convex
    m_pole = np.arccosh(
        (np.roll(G_tilde_0, 1) + np.roll(G_tilde_0, -1)) / (2 * G_tilde_0) )

    return m_pole 

