import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class REDNetGenerator(nn.Module):
    def __init__(self, num_layers=10, num_features=256, in_channels=2, out_channels=2):
        super(REDNetGenerator, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features

        # Encoder
        conv_layers = [
            nn.Sequential(
                nn.Conv2d(in_channels, num_features, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
        ]
        for _ in range(num_layers - 1):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        self.conv_layers = nn.ModuleList(conv_layers)

        # Decoder
        deconv_layers = [
            nn.Sequential(
                nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in range(num_layers - 1)
        ]
        deconv_layers.append(
            nn.ConvTranspose2d(num_features, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.deconv_layers = nn.ModuleList(deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                skip = conv_feats[-(conv_feats_idx + 1)]
                if x.shape == skip.shape:
                    x = x + skip
                conv_feats_idx += 1
                x = self.relu(x)

        x = x + residual
        x = self.relu(x)
        return x


class SRGANDiscriminator(nn.Module):
    def __init__(self, in_channels=2):
        super(SRGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.model(x)


class REDNetGAN(nn.Module):
    def __init__(self):
        super(REDNetGAN, self).__init__()
        self.generator = REDNetGenerator()
        self.discriminator = SRGANDiscriminator()

    def forward(self, x):
        fake = self.generator(x)
        return fake

    def discriminate(self, x):
        return self.discriminator(x)

    def gan_loss(self, fake, real, pred_fake, pred_real, lambda_adv=0.1):
        adv_loss_gen = F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))
        adv_loss_disc = (
            F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real)) +
            F.binary_cross_entropy_with_logits(pred_fake.detach(), torch.zeros_like(pred_fake))
        ) / 2

        recon_loss = F.mse_loss(fake, real)
        total_gan_loss = recon_loss + lambda_adv * adv_loss_gen

        return {
            'loss': total_gan_loss,
            'Reconstruction_Loss': recon_loss.detach(),
            'Adv_Generator': adv_loss_gen.detach(),
            'Adv_Discriminator': adv_loss_disc.detach(),
            'D_loss': adv_loss_disc,
            'G_loss': adv_loss_gen
        }
