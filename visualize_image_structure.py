# -*- coding: utf-8 -*-
import argparse
import math

import numpy as np
import cv2
import matplotlib.pyplot as plt

class Sampler(np.ndarray):
    def __new__(cls, arr):
        return arr.view(cls)

    # wrap, clamp, mirror, border
    def __getitem__(self, key):
#        clamped_key = list(key)
#        for i in range(len(key)):
#            clamped_key[i] = np.clip(key[i], 0, self.shape[i] - 1)
#        return super().__getitem__(tuple(clamped_key))
        is_inside = True
        for i in range(len(key)): is_inside &= (0 <= key[i] < self.shape[i])
        if is_inside: return super().__getitem__(key)
        else: return 0

parser = argparse.ArgumentParser()
parser.add_argument("image_path")
args = parser.parse_args()

color_image = cv2.imread(args.image_path, 1) # color
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
sampler = Sampler(gray_image)

scan_line = 240

display_image = color_image.copy()
cv2.line(display_image, 
    (0, scan_line), (color_image.shape[1], scan_line), 
    (0, 0, 255), 1)
cv2.imshow("Source Image", display_image)
cv2.waitKey(100)


x = list(range(gray_image.shape[1]))
y = gray_image[scan_line, x]
plt.plot(x, y, label='source')

blurred_image = cv2.GaussianBlur(gray_image, (21, 21), 0)
x = list(range(blurred_image.shape[1]))
y = blurred_image[scan_line, x]
plt.plot(x, y, label='blur (wing=10)')
cv2.imshow("blur (wing=10)", blurred_image)
cv2.waitKey(100)

plt.legend()
plt.show()

from mpl_toolkits.mplot3d import Axes3D

x, y = np.meshgrid(range(blurred_image.shape[1]), range(blurred_image.shape[0]))
z = blurred_image[y, x]
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x, y, z)
plt.show()
