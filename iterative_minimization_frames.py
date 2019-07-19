#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import itertools
import sys

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

# initial frame
liv_frame = cv2.imread(f"{args.dataset_path}0.png", 0).astype(np.float32)
liv_frame = cv2.GaussianBlur(liv_frame, (21, 21), 0)
liv_frame = Sampler(liv_frame, BorderAccessMode.CLAMP)

for i in itertools.count(1):
    ref_frame = liv_frame
    liv_frame = cv2.imread(f"{args.dataset_path}{i}.png", 0).astype(np.float32)
    liv_frame = cv2.GaussianBlur(liv_frame, (21, 21), 0)
    liv_frame = Sampler(liv_frame, BorderAccessMode.NONE)

    gen_mat = liealg.sl
    dim_param = gen_mat.shape[0]
    est_mat = np.identity(3, dtype=np.float32)

    depth: np.float64 = 1.0e-20
    best_est = (sys.float_info.max, None) # (error, est_mat)
    for k in range(20):
        print(f"Iteration #{k}")

        #================================================
        # on CPU
        #================================================
        error: np.float128 = 0.
        mat_A = np.zeros((dim_param, dim_param), dtype=np.float128)
        mat_b = np.zeros((dim_param), dtype=np.float128)
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

        x = np.linalg.solve(mat_A.astype(np.float64), mat_b.astype(np.float64))
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

        # Store the best estimation so far
        if error < best_est[0]:
            best_est = (error, est_mat)

    print(f"Best estimation is: \n{best_est[1]}\nwith Error: {best_est[0]}")

    # Display merged image
    synth_colored_frame = cp.empty((ref_frame.shape[0], ref_frame.shape[1], 3), dtype=np.float32)
    dimBlock = (32, 32, 1)
    dimGrid = (
        (ref_frame.shape[1] - 1) // dimBlock[0] + 1, 
        (ref_frame.shape[0] - 1) // dimBlock[1] + 1, 
        1)
    kernel.warp_and_blend_images(dimGrid, dimBlock, (
            cp.asarray(liv_frame), cp.asarray(ref_frame), cp.asarray(best_est[1]), 
            synth_colored_frame, 
            ref_frame.shape[1], ref_frame.shape[0]
        ))
    cp.cuda.Stream.null.synchronize()

    cv2.imshow(f"Synthesized Frame #{k}", synth_colored_frame.get().astype(np.uint8))
    
    key = cv2.waitKey(0)
    if key == ord('q'): break
