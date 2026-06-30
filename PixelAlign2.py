
import torch.nn as nn

class Pixel_align(nn.Module):
    def __init__(self, channels):
        super(Pixel_align, self).__init__()

        self.conv_0 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)

        )

    def forward(self, x_init):
        x = self.conv_0(x_init) + x_init
        out1 = self.conv_0(x) + x
        out = self.conv_0(out1) + out1
        return out + x
