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

def diff_vec(vec):
    vec_pad = np.pad(vec, (1, 1), 'edge')
    return vec_pad[2:] - vec_pad[:-2]

parser = argparse.ArgumentParser()
parser.add_argument("dataset_path")
args = parser.parse_args()

# initial frame
liv_frame_raw = cv2.imread(f"{args.dataset_path}0.png", 0).astype(np.float32)
###liv_frame = cv2.GaussianBlur(liv_frame_raw, (21, 21), 0)
liv_frame = liv_frame_raw
#liv_frame = Sampler(liv_frame, BorderAccessMode.CLAMP)

for i in itertools.count(1):
    ref_frame_raw = liv_frame_raw
    ref_frame = liv_frame
    #ref_frame.border_access_mode = BorderAccessMode.CLAMP
    liv_frame_raw = cv2.imread(f"{args.dataset_path}{i}.png", 0).astype(np.float32)
    ###liv_frame = cv2.GaussianBlur(liv_frame_raw, (21, 21), 0)
    liv_frame = liv_frame_raw
    #liv_frame = Sampler(liv_frame, BorderAccessMode.NONE)

#    synth_colored_frame = cp.empty((ref_frame.shape[0], ref_frame.shape[1], 3), dtype=np.float32)
#    dimBlock = (32, 32, 1)
#    dimGrid = (
#        (ref_frame.shape[1] - 1) // dimBlock[0] + 1, 
#        (ref_frame.shape[0] - 1) // dimBlock[1] + 1, 
#        1)
#    kernel.blend_images(dimGrid, dimBlock, (
#            cp.asarray(liv_frame), cp.asarray(ref_frame), synth_colored_frame, 
#            ref_frame.shape[1], ref_frame.shape[0]
#        ))
#    cp.cuda.Stream.null.synchronize()
#    cv2.imwrite(f"./log/synth_input.png", synth_colored_frame.get().astype(np.uint8))
    cv2.imwrite(f"./log/sub_input.png", (abs(ref_frame-liv_frame)).astype(np.uint8))

    diff_vec(ref_frame[0])
    ref_Iu = np.apply_along_axis(diff_vec, axis=1, arr=ref_frame)
    ref_Iv = np.apply_along_axis(diff_vec, axis=0, arr=ref_frame)

    gen_mat = liealg.sl
    dim_param = gen_mat.shape[0]
    est_mat = np.identity(3, dtype=np.float32)

    depth: np.float64 = 1.0e-20
    error_list = []
    best_est = (sys.float_info.max, None) # (error, est_mat)
    for k in range(20):
        print(f"Iteration #{k}")

        #================================================
        # on CPU
        #================================================
        error: np.float128 = 0.
        mat_A = np.zeros((dim_param, dim_param), dtype=np.float128)
        mat_b = np.zeros((dim_param), dtype=np.float128)
        for (v, u), _ in np.ndenumerate(ref_frame):
            warped_coord = est_mat @ np.array([u, v, 1])
            x, y = warped_coord[:-1] / warped_coord[-1]
            x, y = int(x), int(y)
            if not (0 <= x < liv_frame.shape[1] and 0 <= y < liv_frame.shape[0]): continue
            #if liv_frame[y, x] is None: continue

            #Iu = (ref_frame[v, u + 1] - ref_frame[v, u - 1]) / 2 # 255:-255
            #Iv = (ref_frame[v + 1, u] - ref_frame[v - 1, u]) / 2 # 255:-255
            Iu = ref_Iu[v, u]
            Iv = ref_Iv[v, u]

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

        error_list.append(error)

    print(f"Best estimation is: \n{best_est[1]}\nwith Error: {best_est[0]}")

    #============================================
    # Calculate by feature points as ground truth
    #============================================
    akaze = cv2.AKAZE_create()
    kp0, desc0 = akaze.detectAndCompute(liv_frame_raw, None)
    kp1, desc1 = akaze.detectAndCompute(ref_frame_raw, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc0, desc1)
    good = []
    pts0 = []
    pts1 = []
    for m in matches:
        good.append([m])
        pts1.append(kp1[m.trainIdx].pt)
        pts0.append(kp0[m.queryIdx].pt)
    h_mat, mask = cv2.findHomography(np.array(pts0), np.array(pts1), cv2.LMEDS)
    print(f"Groud Truth is: \n{h_mat}")

    # Show error graph
    plt.xlabel("The number of the iteration")
    plt.ylabel("The amount of the error")
    plt.xticks(range(len(error_list)))
    plt.plot(list(range(len(error_list))), np.array(error_list))
    plt.show()

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
    cv2.imwrite(f"./log/synth_result.png", synth_colored_frame.get().astype(np.uint8))

    cv2.imshow(f"Synthesized Frame #{k}", synth_colored_frame.get().astype(np.uint8))

    sub_image = np.empty(ref_frame.shape, dtype=np.float32)
    warp_mat = best_est[1].flatten()
    for (v, u), val in np.ndenumerate(ref_frame):
        wz = warp_mat[6] * u + warp_mat[7] * v + warp_mat[8];
        wx = int((warp_mat[0] * u + warp_mat[1] * v + warp_mat[2]) / wz);
        wy = int((warp_mat[3] * u + warp_mat[4] * v + warp_mat[5]) / wz);
        if 0 <= wx < liv_frame.shape[1] and 0 <= wy < liv_frame.shape[0]:
            sub_image[v, u] = abs(val - liv_frame[wy, wx])
        else: 
            sub_image[v, u] = val

    cv2.imwrite(f"./log/sub_result.png", sub_image.astype(np.uint8))
    
    key = cv2.waitKey(0)
    if key == ord('q'): break
