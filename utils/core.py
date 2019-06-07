# -*- coding: utf-8 -*-
from enum import Enum, auto

import numpy as np

class BorderAccessMode(Enum):
    CLAMP = auto()
    ZERO = auto()
    NONE = auto()

class Sampler(np.ndarray):
    border_access_mode: BorderAccessMode = BorderAccessMode.CLAMP

    def __new__(cls, arr, border_access_mode):
        return arr.view(cls)

    def __init__(self, arr, border_access_mode):
        self.border_access_mode = border_access_mode

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
