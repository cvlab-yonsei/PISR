from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
import os
from easydict import EasyDict as edict


def _get_default_config():
    c = edict()

    # dataset
    c.data = edict()
    c.data.train = edict()
    c.data.train.params = edict()
    c.data.valid = edict()
    c.data.valid.params = edict()
    c.data.test = edict()
    c.data.test.params = edict()

    # model
    c.model = edict()
    c.model.params = edict()
    c.teacher_model = edict()
    c.teacher_model.params = edict()
    c.student_model = edict()
    c.student_model.params = edict()
    c.ref_model = edict()
    c.ref_model.params = edict()


    # train
    c.train = edict()
    c.train.params = edict()

    # evaluation
    c.eval = edict()
    c.eval.params = edict()

    # optimizer
    c.optimizer = edict()
    c.optimizer.params = edict()

    # scheduler
    c.scheduler = edict()
    c.scheduler.params = edict()

    # losses
    c.loss = edict()
    c.loss.name = None
    c.loss.params = edict()

    # transforms
    c.transform = edict()
    c.transform.name = None
    c.transform.params = edict()

    c.visualizer = edict()
    c.visualizer.name = None
    c.visualizer.params = edict()


    return c


def _merge_config(src, dst):
    if not isinstance(src, edict):
        return

    for k, v in src.items():
        if isinstance(v, edict):
            print('*'*25)
            print(k, v)
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v


def load(config_path):
    with open(config_path, 'r') as fid:
        yaml_config = edict(yaml.load(fid))
    with open(yaml_config.base_config, 'r') as fid:
        base_config = edict(yaml.load(fid))
    config = _get_default_config()
    _merge_config(base_config, config)
    _merge_config(yaml_config, config)

    set_model_weight_dirs(config)

    return config


def set_model_weight_dirs(config):

    if 'name' in config.teacher_model:
        teacher_dir = os.path.join(config.train.dir, config.teacher_model.name)
        config.train.teacher_dir = teacher_dir + config.train.teacher_dir
    if 'name' in config.student_model:
        student_dir = os.path.join(config.train.dir, config.student_model.name)
        config.train.student_dir = student_dir + config.train.student_dir
    if 'name' in config.ref_model:
        ref_dir = os.path.join(config.train.dir, config.ref_model.name)
        config.train.ref_dir = ref_dir + config.train.ref_dir



