#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

import numpy as np
import cv2

frame = np.zeros((480, 640, 3), dtype=np.float32)

for v in range(frame.shape[0]):
    for u in range(frame.shape[1]):
        val = 255 * (u + v) \
                / (frame.shape[1] + frame.shape[0])
        frame[v, u] = [val] * 3

#theta = math.radians(3.0)
#proj_mat = np.array([
#        [1, 0, frame.shape[1] / 2], 
#        [0, 1, frame.shape[0] / 2], 
#        [0, 0, 1], 
#    ]) @ np.array([
#        [math.cos(theta), -math.sin(theta), 0],
#        [math.sin(theta),  math.cos(theta), 0], 
#        [0, 0, 1], 
#    ]) @ np.array([
#        [1, 0, -frame.shape[1] / 2], 
#        [0, 1, -frame.shape[0] / 2], 
#        [0, 0, 1], 
#    ])
warp_mat = np.array([
        [1., 0., 5.], 
        [0., 1., 0.], 
        [0., 0., 1.], 
    ])


warped_frame = cv2.warpPerspective(frame, warp_mat, (frame.shape[1], frame.shape[0]))

synth_frame = np.empty_like(frame)
for v in range(frame.shape[0]):
    for u in range(frame.shape[1]):
            max_val = max(frame[v, u, 0], warped_frame[v, u, 0])
            synth_frame[v, u] = [max_val, max_val, max_val]


cv2.imshow("Frame", frame.astype(np.uint8))
cv2.imshow("Warped Frame", warped_frame.astype(np.uint8))
cv2.imshow("Synth Frame", synth_frame.astype(np.uint8))
cv2.waitKey(0)

#cv2.imwrite("./data/slope0.png", frame.astype(np.uint8))
#cv2.imwrite("./data/slope1.png", warped_frame.astype(np.uint8))
#exit(0)

# visualize structure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

scan_line = 240

x = list(range(frame.shape[1]))
y = frame[scan_line, x, 0]
plt.plot(x, y, label="source", alpha=0.5)

x = list(range(warped_frame.shape[1]))
y = warped_frame[scan_line, x, 0]
plt.plot(x, y, label="warped", alpha=0.5)

plt.legend()
plt.show()

x, y = np.meshgrid(range(frame.shape[1]), range(frame.shape[0]))
z = synth_frame[y, x, 0]
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(x, y, z)
plt.show()
