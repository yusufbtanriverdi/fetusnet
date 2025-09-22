import torch
import torch.nn as nn
import torch.nn.functional as F

class ResDoubleConv(nn.Module):
    """ BN -> ReLU -> Conv3D -> BN -> ReLU -> Conv3D """

    def __init__(self, in_channels, out_channels, num_groups=1):

        super().__init__()
        self.double_conv = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.skip = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        )

    def forward(self, x):
        return self.double_conv(x) + self.skip(x)


class ResDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            ResDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class ResUp(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=False):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = ResDoubleConv(in_channels + in_channels // 2, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=3, stride=2,
                                         padding=1, output_padding=1)
            self.conv = ResDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
