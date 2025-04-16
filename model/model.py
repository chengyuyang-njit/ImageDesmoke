import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from base.base_model import UNet

class LearnableWienerLayer(nn.Module):
    def __init__(self, channels=3, kernel_size=5):
        super(LearnableWienerLayer, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Depthwise convolution: one kernel per channel
        self.filter = nn.Conv2d(
            channels, channels, kernel_size=kernel_size,
            groups=channels, bias=False, padding=self.padding
        )

        # Initialize filter to Gaussian blur (Wiener-like)
        for name, param in self.filter.named_parameters():
            if "weight" in name:
                with torch.no_grad():
                    param.copy_(self._init_gaussian_kernel(channels, kernel_size))

        # Learnable noise level (denominator for Wiener-like control)
        self.noise_power = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        # Smooth signal
        signal = self.filter(x)

        # Simulate Wiener-like effect: weighted mix
        # output = signal * (signal^2 / (signal^2 + noise_power))
        eps = 1e-6
        power = signal ** 2
        wiener_response = power / (power + self.noise_power.view(1, -1, 1, 1) + eps)
        return signal * wiener_response

    def _init_gaussian_kernel(self, channels, kernel_size):
        import numpy as np
        import math

        def gaussian_2d(size, sigma=1.0):
            ax = np.arange(-size // 2 + 1., size // 2 + 1.)
            xx, yy = np.meshgrid(ax, ax)
            kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
            return kernel / np.sum(kernel)

        kernel = gaussian_2d(kernel_size, sigma=1.0)
        kernel = torch.from_numpy(kernel).float()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        return kernel.repeat(channels, 1, 1, 1)


class UNetWithWiener(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetWithWiener, self).__init__()
        self.wiener = LearnableWienerLayer(out_channels)

        self.unet = UNet(in_channels=in_channels, out_channels=out_channels)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.wiener(x)
        x = self.unet(x)
        # x = self.unet(x)
        # x = self.wiener(x)
        return x
    

# model = LearnableWienerLayer()
# model = model.cuda()
# summary(model, input_size = (3, 700, 350))

