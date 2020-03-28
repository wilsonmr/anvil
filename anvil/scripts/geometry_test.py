#!/usr/bin/env python
"""
geometry_test.py

Simple program to test the implementation of the geometry

"""
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch

from anvil.geometry import Geometry2D

L = 2 # very small system

def main():
    """ performs simple test of the geometry setup
        the examples in the original version of geometry.py
        are reproduced
    """
    geometry=Geometry2D(L) #set up a 4x4 hard-wired checkerboard geometry
    print(geometry.checkerboard)    
    print(geometry.splitcart)
    print(geometry.splitlexi)

    state_split = torch.tensor([0, 3, 1, 2])
    shift = geometry.get_shift()
    print(state_split[shift])
    shift = geometry.get_shift(shifts=((1, 1),), dims=((0, 1),))  
    print(state_split[shift])

if __name__ == "__main__":
    main()
