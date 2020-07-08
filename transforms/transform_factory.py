from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os

import numpy as np
import torch
import random


def get_transform(config):

    if config.transform.name is None:
        return None
    else:
        f = globals().get(config.transform.name)
        return f(**config.transform.params)


def base_transform():

    def transform_fn(lr, hr, hflip=None, vflip=None, rot90=None, count=None):

        if hflip is None and vflip is None and rot90 is None and count is None:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            rot90 = random.random() < 0.5
            count = 1 if random.random() < 0.5 else 3

        if hflip:
            lr = torch.flip(lr, dims=(0,))
            hr = torch.flip(hr, dims=(0,))
        if vflip:
            lr = torch.flip(lr, dims=(1,))
            hr = torch.flip(hr, dims=(1,))
        if rot90:
            lr = torch.rot90(lr, count, dims=(1, 2))
            hr = torch.rot90(hr, count, dims=(1, 2))
        return lr, hr

    return transform_fn
    #return None


