import math
from math import exp
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def get_psnr(sr, hr, scale, rgb_range, benchmark=False):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)


# reference : https://github.com/Po-Hsun-Su/pytorch-ssim
def get_ssim(pred, hr, scale, rgb_range, benchmark=False):
    if benchmark:
        shave = scale
    else:
        shave = scale + 6
    pred_channel = pred.shape[1]
    if pred_channel > 1:
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = pred.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        pred = pred.mul(convert).sum(dim=1, keepdim=True)
        hr = hr.mul(convert).sum(dim=1, keepdim=True)
    ssim_loss = SSIM(window_size = 11, size_average=True, channel=1)
    shave = scale
    pred_shaved = pred[..., shave:-shave, shave:-shave]
    hr_shaved = hr[..., shave:-shave, shave:-shave]
    ssim = ssim_loss(pred_shaved, hr_shaved)
    return ssim.data.item()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = 0, groups = channel)
    mu2 = F.conv2d(img2, window, padding = 0, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = 0, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = 0, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = 0, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=1):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
