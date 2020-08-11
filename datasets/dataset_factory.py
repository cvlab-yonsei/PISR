from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import torch.utils.data as utils

from torch.utils.data import dataloader
from .dataset import DIV2K, Benchmark, T91, General100, BSDS200
from .dataset import DIV2KRAM, T91RAM, BSDS200RAM


def get_train_dataset(config, transform=None):
    scale = config.data.scale
    n_colors = config.data.n_colors
    rgb_range = config.data.rgb_range
    batch_size = config.train.batch_size
    datasets = []
    for train_dict in config.data.train:
        name = train_dict.name
        params = train_dict.params
        f = globals().get(name)
        datasets.append(f(**params,
                          scale=scale, n_colors=n_colors, rgb_range=rgb_range,
                          batch_size=batch_size,
                          name=name, train=True, transform=transform,
                         ))
    dataset = ConcatDataset(datasets)

    return dataset


def get_train_dataloader(config, transform=None):
    num_workers = config.data.num_workers
    pin_memory = config.data.pin_memory
    dataset = get_train_dataset(config, transform)
    batch_size = config.train.batch_size
    train_dataloader = dataloader.DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    return train_dataloader


def get_valid_dataset(config, transform=None):
    scale = config.data.scale
    n_colors = config.data.n_colors
    rgb_range = config.data.rgb_range
    batch_size = 1
    datasets = []
    for valid_dict in config.data.valid:
        name = valid_dict.name
        params = valid_dict.params
        f = globals().get(name)
        datasets.append(f(**params,
                          scale=scale, n_colors=n_colors, rgb_range=rgb_range,
                          batch_size=batch_size,
                          name=name, train=False,
                         ))
    dataset = ConcatDataset(datasets)

    return dataset


def get_valid_dataloader(config, transform=None):
    num_workers = config.data.num_workers
    dataset = get_valid_dataset(config, transform)
    batch_size = 1
    valid_dataloader = dataloader.DataLoader(dataset,
                              shuffle=False,
                              batch_size=batch_size,
                              drop_last=False,
                              num_workers=num_workers,
                              pin_memory=True)
    return valid_dataloader


def get_test_dataset(config, transform=None):
    scale = config.data.scale
    n_colors = config.data.n_colors
    rgb_range = config.data.rgb_range
    batch_size = 1
    dataset_dict = dict()
    for test_dict in config.data.test:
        name = test_dict.name
        params = test_dict.params
        dataset_dict[name] = Benchmark(**params,
                          scale=scale, n_colors=n_colors,
                          rgb_range=rgb_range, batch_size=batch_size,
                          name=name)
    return dataset_dict


def get_test_dataloader(config, transform=None):
    num_workers = config.data.num_workers
    dataset_dict = get_test_dataset(config, transform)
    test_dataloader_dict = dict()
    for name, dataset in dataset_dict.items():
        test_dataloader_dict[name] = dataloader.DataLoader(dataset,
                                      shuffle=False,
                                      batch_size=1,
                                      drop_last=False,
                                      pin_memory=True,
                                      num_workers=num_workers)
    return test_dataloader_dict

