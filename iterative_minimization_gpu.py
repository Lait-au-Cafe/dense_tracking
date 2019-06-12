#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import numpy as np
import cupy as cp
import cv2
import matplotlib.pyplot as plt

from utils import Sampler, BorderAccessMode
from utils import liealg
from utils import kernel

parser = argparse.ArgumentParser()
parser.add_argument("dataset_path")
args = parser.parse_args()

ref_frame = cv2.imread(f"{args.dataset_path}0.png", 0).astype(np.float32)
ref_frame = cv2.GaussianBlur(ref_frame, (21, 21), 0)
ref_frame = Sampler(ref_frame, BorderAccessMode.CLAMP)
liv_frame = cv2.imread(f"{args.dataset_path}1.png", 0).astype(np.float32)
liv_frame = cv2.GaussianBlur(liv_frame, (21, 21), 0)
liv_frame = Sampler(liv_frame, BorderAccessMode.NONE)

# Display merged image
synth_colored_frame = cp.empty((ref_frame.shape[0], ref_frame.shape[1], 3), dtype=np.float32)
dimBlock = (32, 32, 1)
dimGrid = (
    (ref_frame.shape[1] - 1) // dimBlock[0] + 1, 
    (ref_frame.shape[0] - 1) // dimBlock[1] + 1, 
    1)
kernel.blend_images(dimGrid, dimBlock, (
        cp.asarray(liv_frame), cp.asarray(ref_frame), synth_colored_frame, 
        ref_frame.shape[1], ref_frame.shape[0]
    ))

cv2.imshow(f"Reference Frame ", ref_frame.astype(np.uint8))
cv2.imshow(f"Live Frame ", liv_frame.astype(np.uint8))
cv2.imshow(f"Synthesized Frame", synth_colored_frame.get().astype(np.uint8))

key =cv2.waitKey(100)

#================================================
# Iterative Minimization (desirably). 
#================================================

d_gen_mat = cp.asarray(liealg.sl)
dim_param = d_gen_mat.shape[0]
d_est_mat = cp.identity(3, dtype=np.float32)
for k in range(1000000):
    print(f"Iteration #{k}")

    #================================================
    # on GPU
    #================================================
    d_error = cp.array([0.], dtype=np.float64)
    d_mat_A = cp.zeros((dim_param, dim_param), dtype=np.float64)
    d_mat_b = cp.zeros((dim_param), dtype=np.float64)

    kernel.step_minimization(dimGrid, dimBlock, (
            cp.asarray(liv_frame), cp.asarray(ref_frame), 
            ref_frame.shape[1], ref_frame.shape[0], 
            d_est_mat.astype(np.float32), d_gen_mat, dim_param, 
            d_mat_A, d_mat_b, d_error
        ))

    np.set_printoptions(linewidth=100000)
    print("d_error")
    print(d_error.get())
    print("d_mat_A")
    print(d_mat_A.get())
    print("d_mat_b")
    print(d_mat_b.get())

    d_x = cp.linalg.solve(d_mat_A, d_mat_b)
    d_x *= -1 # invert
    print("d_x")
    print(d_x.get())

    d_alg_mat = cp.zeros((3, 3), dtype=np.float64)
    for i in range(dim_param):
        d_alg_mat += d_x[i] * d_gen_mat[i]

    d_grp_mat = cp.identity(3) + d_alg_mat

    d_est_mat = d_est_mat @ d_grp_mat
    print("est_mat")
    print(d_est_mat.get())


    # Display merged image
    synth_colored_frame = cp.empty((ref_frame.shape[0], ref_frame.shape[1], 3), dtype=np.float32)
    dimBlock = (32, 32, 1)
    dimGrid = (
        (ref_frame.shape[1] - 1) // dimBlock[0] + 1, 
        (ref_frame.shape[0] - 1) // dimBlock[1] + 1, 
        1)
    kernel.warp_and_blend_images(dimGrid, dimBlock, (
            cp.asarray(liv_frame), cp.asarray(ref_frame), d_est_mat, 
            synth_colored_frame, 
            ref_frame.shape[1], ref_frame.shape[0]
        ))

    cv2.imshow(f"Synthesized Frame #{k}", synth_colored_frame.get().astype(np.uint8))
    
    key =cv2.waitKey(0)
    if key == ord('q'): break
