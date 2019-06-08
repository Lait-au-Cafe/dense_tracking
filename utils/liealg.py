# -*- coding: utf-8 -*-
import numpy as np

# Special Linear Group [Perspective Matrix]
sl = np.array([
    [   [0, 0, 1], 
        [0, 0, 0], 
        [0, 0, 0], ], 

    [   [0, 0, 0], 
        [0, 0, 1], 
        [0, 0, 0], ], 

    [   [0, 1, 0], 
        [0, 0, 0], 
        [0, 0, 0], ], 

    [   [0, 0, 0], 
        [1, 0, 0], 
        [0, 0, 0], ], 

    [   [1, 0, 0], 
        [0,-1, 0], 
        [0, 0, 0], ], 

    [   [0, 0, 0], 
        [0,-1, 0], 
        [0, 0, 1], ], 

    [   [0, 0, 0], 
        [0, 0, 0], 
        [1, 0, 0], ], 

    [   [0, 0, 0], 
        [0, 0, 0], 
        [0, 1, 0], ], 
    ], dtype=np.float64)

# Special Euclidian Group [Transform Matrix]
se = np.array([
    [   [0, 0, 1], 
        [0, 0, 0], 
        [0, 0, 0], ], 

    [   [0, 0, 0], 
        [0, 0, 1], 
        [0, 0, 0], ], 

    [   [0,-1, 0], 
        [1, 0, 0], 
        [0, 0, 0], ], 
    ], dtype=np.float64)

# Affine Group [Affine Matrix]
affine = np.array([
    [   [0, 0, 1], 
        [0, 0, 0], 
        [0, 0, 0], ], 

    [   [0, 0, 0], 
        [0, 0, 1], 
        [0, 0, 0], ], 

    [   [0,-1, 0], 
        [1, 0, 0], 
        [0, 0, 0], ], 

    [   [0, 1, 0], 
        [1, 0, 0], 
        [0, 0, 0], ], 

    [   [1, 0, 0], 
        [0, 0, 0], 
        [0, 0, 0], ], 

    [   [0, 0, 0], 
        [0, 1, 0], 
        [0, 0, 0], ], 
    ], dtype=np.float64)
