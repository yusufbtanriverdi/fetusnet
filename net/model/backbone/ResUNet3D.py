import torch
import torch.nn as nn
from net.model.modules.resnet import *

class ResUNet3D(nn.Module):
    def __init__(self, input_channels, output_channels, base_features, coord_reg=False):
        """
        3D Residual U-Net for segmentation or heatmap regression tasks.

        Parameters:
        - input_channels: int, Number of input channels (1 for grayscale images).
        - output_channels: int, Number of output channels (1 for binary mask, or N for heatmap regression with N landmarks).
        - base_features: int, Number of base feature channels. Controls the depth of intermediate layers.
        - coord_reg: bool, If True, adds a dense layer for landmark coordinate regression.
        s
        Returns:
        - output: torch.Tensor, Segmentation mask or heatmaps.
        - coord_output: torch.Tensor, (if coord_reg is True) Landmark coordinates of shape (batch_size, output_channels, 3).

        """
        
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_features = base_features

        # Initial convolutional layer with skip connection
        self.initial_block = nn.Sequential(
            nn.Conv3d(input_channels, base_features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=base_features),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(base_features, base_features, kernel_size=3, stride=1, padding=1)
        )
        self.initial_skip = nn.Conv3d(input_channels, base_features, kernel_size=3, stride=1, padding=1)

        # Encoder blocks with downsampling
        self.encoder_block1 = ResDown(base_features, 2 * base_features)
        self.encoder_block2 = ResDown(2 * base_features, 4 * base_features)
        self.encoder_block3 = ResDown(4 * base_features, 8 * base_features)
        self.bottleneck = ResDown(8 * base_features, 16 * base_features)

        # Decoder blocks with upsampling
        self.decoder_block1 = ResUp(16 * base_features, 8 * base_features)
        self.decoder_block2 = ResUp(8 * base_features, 4 * base_features)
        self.decoder_block3 = ResUp(4 * base_features, 2 * base_features)
        self.decoder_block4 = ResUp(2 * base_features, base_features)

        # Output layer
        self.output_layer = Out(base_features, output_channels)

        self.coord_reg = coord_reg
        if coord_reg:
            # Dense layer for landmark regression
            self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc = nn.Linear(512, 3 * output_channels)

    def __str__(self):
        return f"ResUNet3D(input_channels={self.input_channels}, output_channels={self.output_channels}, base_features={self.base_features})"

    def forward(self, x):
        # Initial block with skip connection
        x_initial = self.initial_block(x) + self.initial_skip(x)

        # Encoder path
        x_enc1 = self.encoder_block1(x_initial)     # Downsample to 2x base_features
        x_enc2 = self.encoder_block2(x_enc1)        # Downsample to 4x base_features
        x_enc3 = self.encoder_block3(x_enc2)        # Downsample to 8x base_features
        bottleneck_output = self.bottleneck(x_enc3) # Bottom layer at 16x base_features

        # Decoder path with skip connections
        x_dec1 = self.decoder_block1(bottleneck_output, x_enc3)
        x_dec2 = self.decoder_block2(x_dec1, x_enc2)
        x_dec3 = self.decoder_block3(x_dec2, x_enc1)
        x_dec4 = self.decoder_block4(x_dec3, x_initial)

        # Output layer
        output = self.output_layer(x_dec4)

        if self.coord_reg:
            pooled_features = self.global_avg_pool(bottleneck_output)
            pooled_features = torch.flatten(pooled_features, 1)
            coord_output = self.fc(pooled_features).view(-1, self.output_channels, 3)
            return output, coord_output
        
        else:
            return output
