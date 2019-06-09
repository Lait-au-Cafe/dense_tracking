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

gen_mat = liealg.sl
dim_param = gen_mat.shape[0]
est_mat = np.identity(3)
depth: np.float64 = 1.0e-20
for k in range(5):
    print(f"Iteration #{k}")
    error: np.float64 = 0.

    mat_A = np.zeros((dim_param, dim_param), dtype=np.float64)
    mat_b = np.zeros((dim_param), dtype=np.float64)
    for v in range(ref_frame.shape[0]):
        for u in range(ref_frame.shape[1]):
            warped_coord = est_mat @ np.array([u, v, 1])
            x, y = warped_coord[:-1] / warped_coord[-1]
            x, y = int(x), int(y)
            if liv_frame[y, x] is None: continue

            Iu = (ref_frame[v, u + 1] - ref_frame[v, u - 1]) / 2 # 255:-255
            Iv = (ref_frame[v + 1, u] - ref_frame[v - 1, u]) / 2 # 255:-255

            param = np.array([-Iu, -Iv, (u * Iu + v * Iv) / depth]) \
                        @ gen_mat \
                        @ np.array([u, v, 1])

            mat_A +=  param.reshape(-1, 1) * param
            mat_b += -param * (liv_frame[y, x] - ref_frame[v, u])
            error += 0.5 * (liv_frame[y, x] - ref_frame[v, u]) ** 2

    print("error")
    print(error)

    print("mat_A")
    print(mat_A)
    print("mat_b")
    print(mat_b)
    x = np.linalg.solve(mat_A, mat_b)
    x *= -1 # invert
    print("x")
    print(x)

    alg_mat = np.zeros((3, 3), dtype=np.float64)
    for i in range(dim_param):
        alg_mat += x[i] * gen_mat[i]

    grp_mat = np.identity(3) + alg_mat

    est_mat = est_mat @ grp_mat
    print("est_mat")
    print(est_mat)


    # Display merged image
    synth_colored_frame = cp.empty((ref_frame.shape[0], ref_frame.shape[1], 3), dtype=np.float32)
    dimBlock = (32, 32, 1)
    dimGrid = (
        (ref_frame.shape[1] - 1) // dimBlock[0] + 1, 
        (ref_frame.shape[0] - 1) // dimBlock[1] + 1, 
        1)
    kernel.warp_and_blend_images(dimGrid, dimBlock, (
            cp.asarray(liv_frame), cp.asarray(ref_frame), cp.asarray(est_mat), 
            synth_colored_frame, 
            ref_frame.shape[1], ref_frame.shape[0]
        ))
    cp.cuda.Stream.null.synchronize()

    cv2.imshow(f"Synthesized Frame #{k}", synth_colored_frame.get().astype(np.uint8))
    
    key =cv2.waitKey(0)
    if key == ord('q'): break
