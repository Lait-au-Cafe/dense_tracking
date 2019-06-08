# -*- coding: utf-8 -*-
import argparse
from enum import Enum, auto

import numpy as np
import cv2
import matplotlib.pyplot as plt

class BorderAccessMode(Enum):
    CLAMP = auto()
    ZERO = auto()
    NONE = auto()

class Sampler(np.ndarray):
    border_access_mode: BorderAccessMode = BorderAccessMode.CLAMP

    def __new__(cls, arr):
        return arr.view(cls)

    # wrap, clamp, mirror, border
    def __getitem__(self, key):
        if self.border_access_mode == BorderAccessMode.CLAMP:
            clamped_key = list(key)
            for i in range(len(key)):
                clamped_key[i] = np.clip(key[i], 0, self.shape[i] - 1)

            return self.data[tuple(clamped_key)]

        elif self.border_access_mode == BorderAccessMode.ZERO:
            is_inside = True
            for i in range(len(key)): is_inside &= (0 <= key[i] < self.shape[i])
            if is_inside: return super().__getitem__(key)
            else: return 0

        elif self.border_access_mode == BorderAccessMode.NONE:
            is_inside = True
            for i in range(len(key)): is_inside &= (0 <= key[i] < self.shape[i])
            if is_inside: return super().__getitem__(key)
            else: return None


parser = argparse.ArgumentParser()
parser.add_argument("image_path")
args = parser.parse_args()

ref_frame = cv2.imread(f"{args.image_path}0.png", 0).astype(np.float32)
ref_frame = cv2.GaussianBlur(ref_frame, (21, 21), 0)
ref_frame = Sampler(ref_frame)
liv_frame = cv2.imread(f"{args.image_path}1.png", 0).astype(np.float32)
liv_frame = cv2.GaussianBlur(liv_frame, (21, 21), 0)
liv_frame = Sampler(liv_frame)
print(ref_frame.shape)

outputs = []
for i in range(8):
    outputs.append(np.empty_like(ref_frame.data))

mat_A = np.zeros((8, 8), dtype=np.float64)
mat_b = np.zeros((8), dtype=np.float64)
for v in range(ref_frame.shape[0]):
    for u in range(ref_frame.shape[1]):
        Iu = ref_frame[v, u + 1] - ref_frame[v, u - 1] # 1:-1
        Iv = ref_frame[v + 1, u] - ref_frame[v - 1, u] # 1:-1

        param = np.array([0]*8)
        param[0] = Iu
        param[1] = Iv
        param[2] = v * Iu
        param[3] = u * Iv
        param[4] = u * Iu - v * Iv
        param[5] = -u * Iu - 2 * v * Iv
        param[6] = -u * (u * Iu + v * Iv)
        param[7] = -v * (u * Iu + v * Iv)

        outputs[0][v, u] = param[0]
        outputs[1][v, u] = param[1]
        outputs[2][v, u] = param[2]
        outputs[3][v, u] = param[3]
        outputs[4][v, u] = param[4]
        outputs[5][v, u] = param[5]
        outputs[6][v, u] = param[6]
        outputs[7][v, u] = param[7]

        mat_A +=  param.reshape(-1, 1) * param
        mat_b += -param * (liv_frame[v, u] - ref_frame[v, u])

# Display parameters
fig = plt.figure(figsize=(18, 36))
axs = []
for i in range(8):
    axs.append(fig.add_subplot(4, 2, i + 1))
    axs[i].imshow(outputs[i], cmap='cool')
    axs[i].set_title(f"Param {i}")

fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.)
plt.show()

print(mat_A)
print()
print(mat_b)
print()

x = np.linalg.solve(mat_A, mat_b)
print(x)
print()

x *= -1
proj_mat = np.array([
    [1 + x[4], x[2], x[0]], 
    [x[3], 1 - x[4] - x[5], x[1]], 
    [x[6], x[7], 1 + x[5]]
    ])
print(proj_mat)
print()

liv_frame.border_access_mode = BorderAccessMode.NONE

synth_frame = np.empty_like(ref_frame)
for v in range(ref_frame.shape[0]):
    for u in range(ref_frame.shape[1]):
        warped_coord = proj_mat @ np.array([u, v, 1])
        x, y = warped_coord[:-1] / warped_coord[-1]
        x, y = int(x), int(y)

        synth_frame[v, u] = liv_frame[y, x] or ref_frame[v, u]

#cv2.imshow("Synthesized Frame", synth_frame.astype(np.uint8))
#cv2.waitKey(0)
