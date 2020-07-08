import torch.nn as nn
from collections import OrderedDict
from .encoder import get_encoder
from .base import BaseNet


class FSRCNN(nn.Module):
   def __init__(self, scale, n_colors, d=56, s=12, m=4, fsrcnn_weight_init=False):
       super(FSRCNN, self).__init__()
       self.scale = scale
       self.feature_extraction = []
       self.feature_extraction.append(nn.Sequential(
           nn.Conv2d(in_channels=n_colors,
                     out_channels=d, kernel_size=5, stride=1, padding=2),
           nn.PReLU()))
       self.shrinking = []
       self.shrinking.append(nn.Sequential(
           nn.Conv2d(in_channels=d, out_channels=s,
                     kernel_size=1, stride=1, padding=0),
           nn.PReLU()))
       self.mapping = []
       for _ in range(m):
           self.mapping.append(nn.Sequential(
               nn.Conv2d(in_channels=s, out_channels=s,
                         kernel_size=3, stride=1, padding=1),
               nn.PReLU()))
       self.expanding = []
       self.expanding.append(nn.Sequential(
           nn.Conv2d(in_channels=s, out_channels=d,
                     kernel_size=1, stride=1, padding=0),
           nn.PReLU()))
       self.last_layer = []
       self.last_layer.append(nn.Sequential(
           nn.ConvTranspose2d(d, n_colors, kernel_size=9, stride=scale, padding=9//2,
                              output_padding=scale-1))
       )
       self.network = nn.Sequential(
           OrderedDict([
               ('feature_extraction', nn.Sequential(*self.feature_extraction)),
               ('shrinking', nn.Sequential(*self.shrinking)),
               ('mapping', nn.Sequential(*self.mapping)),
               ('expanding', nn.Sequential(*self.expanding)),
               ('last_layer', nn.Sequential(*self.last_layer)),
           ]))

       if fsrcnn_weight_init:
           self.fsrcnn_weight_init()


   def fsrcnn_weight_init(self, mean=0.0, std=0.001):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(mean, std)
                if m.bias is not None:
                    m.bias.data.zero_()

   def forward(self, x):
       return self.network(x)


class FSRCNNAutoencoder(FSRCNN):
    def __init__(self, scale, n_colors, d=56, s=12, m=4, k=1, encoder='inv_fsrcnn'):
        super(FSRCNNAutoencoder, self).__init__(scale, n_colors, d, s, m)

        self.encoder = get_encoder(encoder, scale=scale, d=d, s=s, k=k, n_colors=n_colors)
        self.feature_extraction = []
        self.feature_extraction.append(nn.Sequential(
            nn.Conv2d(in_channels=n_colors*k,
                        out_channels=d, kernel_size=5, stride=1, padding=2),
            nn.PReLU()))
        self.network = nn.Sequential(
            OrderedDict([
                ('encoder', nn.Sequential(*self.encoder)),
                ('feature_extraction', nn.Sequential(*self.feature_extraction)),
                ('shrinking', nn.Sequential(*self.shrinking)),
                ('mapping', nn.Sequential(*self.mapping)),
                ('expanding', nn.Sequential(*self.expanding)),
                ('last_layer', nn.Sequential(*self.last_layer)),
            ]))


    def forward(self, x):
        return self.network(x)



class FSRCNNStudent(BaseNet):
    def __init__(self, scale, n_colors, d=56, s=12, m=4, vid_info=None,
                modules_to_freeze=None, initialize_from=None,
                modules_to_initialize=None, fsrcnn_weight_init=False):

        super(FSRCNNStudent, self).__init__()
        self.scale = scale
        self.initialize_from = initialize_from
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_initialize = modules_to_initialize
        self.backbone = FSRCNN(scale, n_colors, d, s, m, fsrcnn_weight_init=fsrcnn_weight_init)
        self.vid_info = vid_info if vid_info is not None else []
        self.vid_module_dict = self.get_vid_module_dict()

        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()



    def forward(self, LR, HR=None, teacher_pred_dict=None):
        ret_dict = dict()
        x = LR
        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x
            if layer_name in self.distill_layers:
                mean = self.vid_module_dict._modules[layer_name+'_mean'](x)
                var = self.vid_module_dict._modules[layer_name+'_var'](x)
                ret_dict[layer_name+'_mean'] = mean
                ret_dict[layer_name+'_var'] = var
        hr = x
        ret_dict['hr'] = hr
        return ret_dict


class FSRCNNTeacher(BaseNet):
    def __init__(self, scale, n_colors,  d=56, s=12, m=4, k=1, vid_info=None,
                 modules_to_freeze=None, initialize_from=None, modules_to_initialize=None,
                 encoder='inv_fsrcnn'):
        super(FSRCNNTeacher, self).__init__()

        self.scale = scale
        self.initialize_from = initialize_from
        self.modules_to_initialize = modules_to_initialize
        self.modules_to_freeze = modules_to_freeze
        self.backbone = FSRCNNAutoencoder(scale, n_colors, d, s, m, k, encoder)
        self.vid_info = vid_info if vid_info is not None else []
        self.vid_module_dict = self.get_vid_module_dict()

        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()


    def forward(self, HR, LR=None):
        ret_dict = dict()

        x = HR
        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x
            if layer_name in self.distill_layers:
                mean = self.vid_module_dict._modules[layer_name+'_mean'](x)
                var = self.vid_module_dict._modules[layer_name+'_var'](x)
                ret_dict[layer_name+'_mean'] = mean
                ret_dict[layer_name+'_var'] = var

        hr = x
        ret_dict['hr'] = hr
        return ret_dict



def get_fsrcnn_teacher(scale, n_colors, **kwargs):
    return FSRCNNTeacher(scale, n_colors, **kwargs)


def get_fsrcnn_student(scale, n_colors, **kwargs):
    return FSRCNNStudent(scale, n_colors, **kwargs)



