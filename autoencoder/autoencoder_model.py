import torch
import torch.nn as nn
import math

class REDNetBase(nn.Module):
    def __init__(self, num_layers, num_features=128, in_channels=2, out_channels=2):
        """
        Base class for REDNet variations.
        - num_layers: Number of convolutional & deconvolutional layers (e.g., 5, 10, 15)
        - num_features: Number of feature maps per layer
        - in_channels / out_channels: Number of input/output channels (set to 2)
        """
        super(REDNetBase, self).__init__()
        self.num_layers = num_layers

        # Encoder (convolutions)
        conv_layers = []
        conv_layers.append(nn.Sequential(nn.Conv2d(in_channels, num_features, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))  # Downsample from 416 -> 208
        for _ in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        # Decoder (transpose convolutions)
        deconv_layers = []
        for _ in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.ConvTranspose2d(num_features, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1))  # Upsample back to 416

        self.conv_layers = nn.ModuleList(conv_layers)
        self.deconv_layers = nn.ModuleList(deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x  # Save input for final residual connection

        # Encoder forward pass
        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)  # Store activations for skip connections

        # Decoder forward pass with skip connections
        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                if x.shape == conv_feat.shape:  # Ensure shape compatibility
                    x = x + conv_feat
                conv_feats_idx += 1
                x = self.relu(x)

        x += residual  # Final residual connection
        x = self.relu(x)

        return x


# Define specific REDNet variations
class REDNet10(REDNetBase):
    def __init__(self, num_features=128, in_channels=2, out_channels=2):
        super(REDNet10, self).__init__(num_layers=5, num_features=num_features, in_channels=in_channels, out_channels=out_channels)


class REDNet10_W256(REDNetBase):
    def __init__(self, num_features=256, in_channels=2, out_channels=2):
        super(REDNet10_W256, self).__init__(num_layers=5, num_features=num_features, in_channels=in_channels, out_channels=out_channels)


class REDNet20(REDNetBase):
    def __init__(self, num_features=128, in_channels=2, out_channels=2):
        super(REDNet20, self).__init__(num_layers=10, num_features=num_features, in_channels=in_channels, out_channels=out_channels)


class REDNet30(REDNetBase):
    def __init__(self, num_features=128, in_channels=2, out_channels=2):
        super(REDNet30, self).__init__(num_layers=15, num_features=num_features, in_channels=in_channels, out_channels=out_channels)
