import torch.nn as nn

def get_vid_module_dict(network, layers, homoscedasticities):
    vid_module_dict = {}
    for l, h in zip(layers, homoscedasticities):
        # get out_channels from last conv layer
        for name, p in network._modules[l].named_parameters():
            if len(p.shape) == 4:
                channels = p.shape[0]
        mean, var = get_mean_and_variance(channels, h)
        vid_module_dict[l + '_mean'] = mean
        vid_module_dict[l + '_var'] = var
    return vid_module_dict


def get_mean_and_variance(in_channels, homoscedasticity):
    if homoscedasticity == 'C':
        # variances are same for each channel
        var_out_channels = in_channels
        var_adap_avg_pool = True
    elif homoscedasticity == 'HW':
        # variances are same for each spatial point
        var_out_channels = 1
        var_adap_avg_pool = False
    elif homoscedasticity == 'CHW':
        # variances are all same for this layer
        var_out_channels = 1
        var_adap_avg_pool = True
    elif homoscedasticity == 'None':
        # variances are all different
        var_out_channels = in_channels
        var_adap_avg_pool = False

    mean = get_adaptation_layer(in_channels, in_channels, False)
    var = get_adaptation_layer(in_channels, var_out_channels, var_adap_avg_pool)
    var.add_module(str(len(var)+1), nn.Softplus())

    return mean, var


def get_adaptation_layer(in_channels, out_channels, adap_avg_pool):
    layer = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                    kernel_size=1, stride=1, padding=0),
        nn.PReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=1, stride=1, padding=0 )
    )
    if adap_avg_pool:
        layer.add_module(str(len(layer)+1), nn.AdaptiveAvgPool2d(1))

    return layer
