# -*- coding: utf-8 -*-
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

class Sampler:
    data: np.ndarray

    @property
    def shape(self):
        return self.data.shape

    def __init__(self, data):
        self.data = data

    # wrap, clamp, mirror, border
    def __getitem__(self, key):
        clamped_key = list(key)
        for i in range(len(key)):
            clamped_key[i] = np.clip(key[i], 0, self.shape[i] - 1)

        return self.data[tuple(clamped_key)]

parser = argparse.ArgumentParser()
parser.add_argument("image_path")
args = parser.parse_args()

sampler = Sampler(cv2.imread(args.image_path, 0).astype(np.float32))
print(sampler.shape)

outputs = []
for i in range(8):
    outputs.append(np.empty_like(sampler.data))

for v in range(sampler.shape[0]):
    for u in range(sampler.shape[1]):
        Iu = sampler[v, u + 1] - sampler[v, u - 1] # 1:-1
        Iv = sampler[v + 1, u] - sampler[v - 1, u] # 1:-1

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


fig = plt.figure(figsize=(18, 36))
axs = []
for i in range(8):
    axs.append(fig.add_subplot(4, 2, i + 1))
    axs[i].imshow(outputs[i], cmap='cool')
    axs[i].set_title(f"Param {i}")

fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.)
plt.show()
