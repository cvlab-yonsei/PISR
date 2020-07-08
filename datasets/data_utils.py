import os, sys
import threading
import random

import numpy as np
import skimage.color as sc

import torch
import torch.multiprocessing as multiprocessing


def get_patch(*img_pair, patch_size=96, scale=2):
    ih, iw = img_pair[0].shape[:2]

    tp = patch_size
    ip = tp // scale

    ix = random.randrange(0, iw - ip)
    iy = random.randrange(0, ih - ip)
    tx, ty = scale * ix, scale * iy

    ret = [
        img_pair[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in img_pair[1:]]
    ]

    return ret

def set_channel(*img_pair, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in img_pair]

def np2Tensor(*img_pair, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in img_pair]

def augment(*img_pair, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(a) for a in img_pair]
