#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import Sampler, BorderAccessMode

parser = argparse.ArgumentParser()
parser.add_argument("dataset_path")
args = parser.parse_args()

ref_frame = cv2.imread(f"{args.dataset_path}0.png", 0).astype(np.float32)
ref_frame = cv2.GaussianBlur(ref_frame, (21, 21), 0)
ref_frame = Sampler(ref_frame, BorderAccessMode.CLAMP)
liv_frame = cv2.imread(f"{args.dataset_path}1.png", 0).astype(np.float32)
liv_frame = cv2.GaussianBlur(liv_frame, (21, 21), 0)
liv_frame = Sampler(liv_frame, BorderAccessMode.NONE)

proj_mat = np.identity(3)
weight: np.float32 = 1 / 10000
for k in range(5):
    print(f"Iteration #{k}")

    mat_A = np.zeros((8, 8), dtype=np.float64)
    mat_b = np.zeros((8), dtype=np.float64)
    for v in range(ref_frame.shape[0]):
        for u in range(ref_frame.shape[1]):
            warped_coord = proj_mat @ np.array([u, v, 1])
            x, y = warped_coord[:-1] / warped_coord[-1]
            x, y = int(x), int(y)
            if liv_frame[y, x] is None: 
                print(f"({u}, {v}) => ({x}, {y})")
                break

            Iu = (ref_frame[v, u + 1] - ref_frame[v, u - 1]) / 2 # 255:-255
            Iv = (ref_frame[v + 1, u] - ref_frame[v - 1, u]) / 2 # 255:-255

#            param = np.array([
#                    Iu, 
#                    Iv, 
#                    v * Iu, 
#                    u * Iv, 
#                    u * Iu - v * Iv, 
#                    -u * Iu - 2 * v * Iv, 
#                    -u * (u * Iu + v * Iv), 
#                    -v * (u * Iu + v * Iv), 
#                ])
            param = np.array([
                    Iu, 
                    Iv, 
                    v * Iu, 
                    u * Iv, 
                    u * Iu - v * Iv, 
                    -v * Iv - weight * (u * Iu + v * Iv), 
                    -weight * u * (u * Iu + v * Iv), 
                    -weight * v * (u * Iu + v * Iv), 
                ])

            mat_A +=  param.reshape(-1, 1) * param
            mat_b += -param * (liv_frame[y, x] - ref_frame[v, u])

    print("mat_A")
    print(mat_A)
    print("mat_b")
    print(mat_b)
    x = np.linalg.solve(mat_A, mat_b)
    x *= -1 # invert
    proj_mat = proj_mat @ np.array([
            [1 + x[4], x[2], x[0]], 
            [x[3], 1 - x[4] - x[5], x[1]], 
            [x[6], x[7], 1 + x[5]]
        ])
    print("proj_mat")
    print(proj_mat)


    # Display merged image
    synth_frame = np.empty_like(ref_frame)
    synth_colored_frame = np.empty((ref_frame.shape[0], ref_frame.shape[1], 3), dtype=np.float32)
    warped_frame = np.empty_like(ref_frame)
    for v in range(ref_frame.shape[0]):
        for u in range(ref_frame.shape[1]):
            warped_coord = proj_mat @ np.array([u, v, 1])
            x, y = warped_coord[:-1] / warped_coord[-1]
            x, y = int(x), int(y)

            synth_frame[v, u] = liv_frame[y, x] or ref_frame[v, u]
            synth_colored_frame[v, u] = (liv_frame[y, x] or 0, ref_frame[v, u], 0)
            warped_frame[v, u] = liv_frame[y, x] or 0

    cv2.imshow(f"Reference Frame #{k}", ref_frame.astype(np.uint8))
    cv2.imshow(f"Warped Live Frame #{k}", warped_frame.astype(np.uint8))
    cv2.imshow(f"Synthesized Frame #{k}", synth_colored_frame.astype(np.uint8))
    
    key =cv2.waitKey(0)
    if key == ord('q'): break
