import torch.nn as nn


def get_encoder(name, **params):
    f = globals().get('get_%s_encoder'%name)
    return f(**params)


def get_lcscc_encoder(scale=2, d=56, s=12, k=1, n_colors=1):
    encoder = [nn.Sequential(
        nn.Conv2d(in_channels=n_colors,
                    out_channels=d, kernel_size=5, stride=1, padding=2),
        nn.PReLU(),
        nn.Conv2d(in_channels=d,
                    out_channels=d, kernel_size=5, stride=scale, padding=2),
        nn.PReLU(),
        nn.Conv2d(in_channels=d,
                    out_channels=s, kernel_size=3, stride=1, padding=1),
        nn.PReLU(),
        nn.Conv2d(in_channels=s,
                    out_channels=n_colors*k, kernel_size=3, stride=1, padding=1),
    )]
    return encoder

