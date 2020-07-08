import sys
from collections import OrderedDict
import torch
import torch.nn as nn
sys.path.append('../')
from utils.checkpoint import get_last_checkpoint
from .vid_module import get_vid_module_dict

class BaseNet(nn.Module):

    def __init__(self):
        super(BaseNet, self).__init__()
        self.backbone = None
        self.modules_to_freeze = None
        self.initialize_from = None
        self.modules_to_initialize = None


    def freeze_modules(self):
        for k, m in self.backbone.network._modules.items():
            if k in self.modules_to_freeze:
                for param in m.parameters():
                    param.requires_grad = False
                print('freezing layer: %s'%k)


    def load_pretrained_model(self):

        if type(self.initialize_from) != list:
            self.initialize_from = [self.initialize_from]
            self.modules_to_initialize = [self.modules_to_initialize]

        for init_from, modules_to_init in zip(self.initialize_from, self.modules_to_initialize):
            print(init_from)
            checkpoint = get_last_checkpoint(init_from)
            checkpoint = torch.load(checkpoint)
            new_state_dict = self.state_dict()
            for key in checkpoint['state_dict'].keys():
                for k in key.split('.'):
                    if k in modules_to_init:
                        new_state_dict[key] = checkpoint['state_dict'][key]
                        print('pretrain parameters: %s'%k)
            self.load_state_dict(new_state_dict)


    def get_vid_module_dict(self):
        self.distill_layers = []
        self.homoscedasticities = []
        for s in self.vid_info:
            layer, homoscedasticity = s.split(':')
            self.distill_layers.append(layer)
            self.homoscedasticities.append(homoscedasticity)

        vid_module_dict = get_vid_module_dict(self.backbone.network,
                    self.distill_layers, self.homoscedasticities)
        vid_module_dict = nn.Sequential(
            OrderedDict([(k, v) for k, v in vid_module_dict.items()]))
        return vid_module_dict
