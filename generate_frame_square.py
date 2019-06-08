#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

import numpy as np
import cv2

frame = np.zeros((480, 640, 3), dtype=np.float32)

# rectangle
cv2.rectangle(frame, (250, 200), (500, 400), (255, 255, 255), -1)

# blur
frame = cv2.GaussianBlur(frame, (81, 81), 0)
#frame = cv2.GaussianBlur(frame, (21, 21), 0)

theta = math.radians(3.0)
proj_mat = np.array([
        [1, 0, frame.shape[1] / 2], 
        [0, 1, frame.shape[0] / 2], 
        [0, 0, 1], 
    ]) @ np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta),  math.cos(theta), 0], 
        [0, 0, 1], 
    ]) @ np.array([
        [1, 0, -frame.shape[1] / 2], 
        [0, 1, -frame.shape[0] / 2], 
        [0, 0, 1], 
    ])


warped_frame = cv2.warpPerspective(frame, proj_mat, (frame.shape[1], frame.shape[0]))

synth_frame = np.empty_like(frame)
for v in range(frame.shape[0]):
    for u in range(frame.shape[1]):
            max_val = max(frame[v, u, 0], warped_frame[v, u, 0])
            synth_frame[v, u] = [max_val, max_val, max_val]


cv2.imshow("Frame", frame.astype(np.uint8))
cv2.imshow("Warped Frame", warped_frame.astype(np.uint8))
cv2.imshow("Synth Frame", synth_frame.astype(np.uint8))
cv2.waitKey(0)

#cv2.imwrite("./data/square0.png", frame)
#cv2.imwrite("./data/square1.png", warped_frame)
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
