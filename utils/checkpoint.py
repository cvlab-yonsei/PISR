
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import torch


def get_last_checkpoint(checkpoint_dir):
    checkpoints = [checkpoint
                   for checkpoint in os.listdir(checkpoint_dir)
                   if checkpoint.startswith('epoch_') and
                   checkpoint.endswith('.pth')]
    if checkpoints:
        return os.path.join(checkpoint_dir, list(sorted(checkpoints))[-1])
    return None


def get_initial_checkpoint(config, model_type):
    train_dir = config.train[model_type + '_dir']
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    return get_last_checkpoint(checkpoint_dir)


def get_checkpoint(config, name, model_type='teacher'):
    train_dir = config.train[model_type + '_dir']
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    return os.path.join(checkpoint_dir, name)


def copy_last_n_checkpoints(config, n, name, model_type):

    train_dir = config.train[model_type + '_dir']
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    checkpoints = [checkpoint
                   for checkpoint in os.listdir(checkpoint_dir)
                   if checkpoint.startswith('epoch_') and
                   checkpoint.endswith('.pth')]
    checkpoints = sorted(checkpoints)
    for i, checkpoint in enumerate(checkpoints[-n:]):
        shutil.copyfile(os.path.join(checkpoint_dir, checkpoint),
                        os.path.join(checkpoint_dir, name.format(i)))


def load_checkpoint(model, optimizer, checkpoint, model_type):
    print('load checkpoint from', checkpoint)
    checkpoint = torch.load(checkpoint)
    checkpoint_dict = checkpoint['state_dict']
    model.load_state_dict(checkpoint_dict) #, strict=False)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_dict'])

    step = checkpoint['step'] if 'step' in checkpoint else -1
    last_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else -1

    return last_epoch, step


def save_checkpoint(config, model, optimizer,
                    epoch, step, model_type, weights_dict=None, name=None):
    train_dir = config.train[model_type + '_dir']
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')

    if name:
        checkpoint_path = os.path.join(checkpoint_dir, '{}.pth'.format(name))
    else:
        checkpoint_path = os.path.join(checkpoint_dir,
                                       'epoch_{:04d}.pth'.format(epoch))
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    if weights_dict is None:
        weights_dict = {
          'state_dict': state_dict,
          'optimizer_dict' : optimizer.state_dict(),
          'epoch' : epoch,
          'step' : step,
          }
    torch.save(weights_dict, checkpoint_path)
