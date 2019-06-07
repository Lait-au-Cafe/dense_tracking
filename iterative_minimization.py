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
for k in range(2):
    mat_A = np.zeros((8, 8), dtype=np.float64)
    mat_b = np.zeros((8), dtype=np.float64)
    for v in range(ref_frame.shape[0]):
        for u in range(ref_frame.shape[1]):
            warped_coord = proj_mat @ np.array([u, v, 1])
            x, y = warped_coord[:-1] / warped_coord[-1]
            x, y = int(x), int(y)
            if liv_frame[y, x] is None: break

            Iu = (ref_frame[v, u + 1] - ref_frame[v, u - 1]) / 2 # 255:-255
            Iv = (ref_frame[v + 1, u] - ref_frame[v - 1, u]) / 2 # 255:-255

            param = np.array([0]*8)
            param[0] = Iu
            param[1] = Iv
            param[2] = v * Iu
            param[3] = u * Iv
            param[4] = u * Iu - v * Iv
            param[5] = -u * Iu - 2 * v * Iv
            param[6] = -u * (u * Iu + v * Iv)
            param[7] = -v * (u * Iu + v * Iv)

            mat_A +=  param.reshape(-1, 1) * param
            mat_b += -param * (liv_frame[y, x] - ref_frame[v, u])

    x = np.linalg.solve(mat_A, mat_b)
    x *= -1 # invert
    proj_mat = proj_mat @ np.array([
            [1 + x[4], x[2], x[0]], 
            [x[3], 1 - x[4] - x[5], x[1]], 
            [x[6], x[7], 1 + x[5]]
        ])

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

cv2.imshow("Reference Frame", ref_frame.astype(np.uint8))
cv2.imshow("Warped Live Frame", warped_frame.astype(np.uint8))
cv2.imshow("Synthesized Frame", synth_colored_frame.astype(np.uint8))
cv2.waitKey(0)
