from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from torch.optim.optimizer import Optimizer, required
import torch.optim as optim
import torch


def adam(parameters, betas=(0.9, 0.999), weight_decay=0,
         amsgrad=False, **_):
    if isinstance(betas, str):
        betas = eval(betas)
    return optim.Adam(parameters, betas=betas, weight_decay=weight_decay,
                      amsgrad=amsgrad)


def sgd(parameters, momentum=0.9, weight_decay=0, nesterov=True, **_):
    return optim.SGD(parameters, momentum=momentum, weight_decay=weight_decay,
                   nesterov=nesterov)

def get_optimizer(config, model):

    parameters = []
    if isinstance(config.optimizer.params.lr, list):
        for param_lr in config.optimizer.params.lr:
            param, lr = param_lr.split(':')
            lr = float(lr)
            pdict = dict()
            pdict['params'] = model.backbone.network._modules[param].parameters()
            pdict['lr'] = lr
            parameters.append(pdict)
    else:
        pdict = dict()
        if isinstance(model, list):
            params = []
            for m in model:
                for p in m.parameters():
                    params.append(p)
        else:
            params = model.parameters()

        pdict['params'] = filter(lambda p: p.requires_grad, params)
        pdict['lr'] = config.optimizer.params.lr
        parameters.append(pdict)
    f = globals().get(config.optimizer.name)
    return f(parameters, **config.optimizer.params)
