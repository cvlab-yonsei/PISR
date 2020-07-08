import os
import pickle
import numpy as np
import torch
import torch.utils.data as utils
from .base_dataset import BaseDataset, BaseBenchmarkDataset, BaseDatasetRAM


class DIV2KRAM(BaseDatasetRAM):
    def __init__(self, scale=2, n_colors=1, rgb_range=255,
                 batch_size=128,
                 test_every=None, patch_size=192, augment=True, data_range='1-800',
                 base_dir=None,
                 name='DIV2K', train=True, transform=None,
                ):
        data_range = data_range.split('-')
        begin, end = list(map(lambda x: int(x)-1, data_range))
        super(DIV2KRAM, self).__init__(scale, n_colors, rgb_range, batch_size,
                 test_every, patch_size, augment, base_dir, name, train, transform,
                  begin, end)



class DIV2K(BaseDataset):
    def __init__(self, scale=2, n_colors=1, rgb_range=255,
                 batch_size=128,
                 test_every=None, patch_size=192, augment=True, data_range='1-800',
                 base_dir=None,
                 name='DIV2K', train=True, transform=None
                ):
        data_range = data_range.split('-')
        begin, end = list(map(lambda x: int(x)-1, data_range))
        super(DIV2K, self).__init__(scale, n_colors, rgb_range, batch_size,
                 test_every, patch_size, augment, base_dir, name, train,
                  begin, end)


class T91RAM(BaseDatasetRAM):
    def __init__(self, scale=2, n_colors=1, rgb_range=255,
                 batch_size=128,
                 test_every=None, patch_size=192, augment=True, data_range='1-800',
                 base_dir=None, name='T91', train=True, transform=None,
                ):
        data_range = data_range.split('-')
        begin, end = list(map(lambda x: int(x)-1, data_range))
        super(T91RAM, self).__init__(scale, n_colors, rgb_range, batch_size,
                 test_every, patch_size, augment, base_dir, name, train, transform,
                  begin, end)


class T91(BaseDataset):
    def __init__(self, scale=2, n_colors=1, rgb_range=255,
                 batch_size=128,
                 test_every=None, patch_size=192, augment=True, data_range='1-800',
                 base_dir=None,
                 name='T91', train=True, transform=None,
                ):
        data_range = data_range.split('-')
        begin, end = list(map(lambda x: int(x)-1, data_range))
        super(T91, self).__init__(scale, n_colors, rgb_range, batch_size,
                 test_every, patch_size, augment, base_dir, name, train, transform,
                  begin, end)


class BSDS200(BaseDataset):
    def __init__(self, scale=2, n_colors=1, rgb_range=255,
                 batch_size=128,
                 test_every=None, patch_size=192, augment=True, data_range='1-800',
                 base_dir=None,
                 name='BSDS200', train=True, transform=None,
                ):
        data_range = data_range.split('-')
        begin, end = list(map(lambda x: int(x)-1, data_range))
        super(BSDS200, self).__init__(scale, n_colors, rgb_range, batch_size,
                 test_every, patch_size, augment, base_dir, name, train, transform,
                  begin, end)

class BSDS200RAM(BaseDatasetRAM):
    def __init__(self, scale=2, n_colors=1, rgb_range=255,
                 batch_size=128,
                 test_every=None, patch_size=192, augment=True, data_range='1-800',
                 base_dir=None,
                 name='BSDS200', train=True, transform=None,
                ):
        data_range = data_range.split('-')
        begin, end = list(map(lambda x: int(x)-1, data_range))
        super(BSDS200RAM, self).__init__(scale, n_colors, rgb_range, batch_size,
                 test_every, patch_size, augment, base_dir, name, train, transform,
                  begin, end)


class General100(BaseDataset):
    def __init__(self, scale=2, n_colors=1, rgb_range=255,
                 batch_size=128,
                 test_every=None, patch_size=192, augment=True, data_range='1-800',
                 base_dir=None,
                 name='General100', train=True
                ):
        data_range = data_range.split('-')
        begin, end = list(map(lambda x: int(x)-1, data_range))
        super(General100, self).__init__(scale, n_colors, rgb_range, batch_size,
                 test_every, patch_size, augment, base_dir, name, train,
                  begin, end)

class Benchmark(BaseBenchmarkDataset):
    def __init__(self, scale=2, n_colors=1, rgb_range=255, batch_size=1,
                 base_dir=None, name=''):
        super(Benchmark, self).__init__(
            scale, n_colors, rgb_range, batch_size, base_dir, name
        )

