import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class REDNetVAEGenerator(nn.Module):
    def __init__(self, num_layers=5, num_features=256, in_channels=2, out_channels=2, latent_dim=128):
        super(REDNetVAEGenerator, self).__init__()
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.num_features = num_features

        # Encoder
        conv_layers = [
            nn.Sequential(
                nn.Conv2d(in_channels, num_features, kernel_size=3, stride=2, padding=1),  # 416 -> 208
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

        # Latent bottleneck (global average pool → μ/log_var)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, C, 1, 1]
        self.fc_mu = nn.Linear(num_features, latent_dim)
        self.fc_var = nn.Linear(num_features, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, num_features)

        # Decoder
        deconv_layers = [
            nn.Sequential(
                nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in range(num_layers - 1)
        ]
        deconv_layers.append(
            nn.ConvTranspose2d(num_features, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)  # 208 -> 416
        )
        self.deconv_layers = nn.ModuleList(deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def encode(self, x):
        residual = x
        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        pooled = self.global_pool(x).view(x.size(0), -1)
        mu = self.fc_mu(pooled)
        log_var = self.fc_var(pooled)
        return mu, log_var, conv_feats, residual

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z, conv_feats, residual):
        x = self.decoder_input(z).view(-1, self.num_features, 1, 1)
        x = F.interpolate(x, size=(208, 208), mode='bilinear', align_corners=False)

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

    def forward(self, x):
        mu, log_var, conv_feats, residual = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z, conv_feats, residual)
        return recon, x, mu, log_var

    
    def loss_function(self, *args, target, M_N=1.0, kl_weight=1.0):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        recons_loss = F.mse_loss(recons, target, reduction='mean')

        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        kld_loss = torch.mean(kld)

        # kld_loss = torch.clamp(kld_loss, min=0.0)

        loss = recons_loss + kl_weight * M_N * kld_loss

        return {
            'loss': loss,
            'Reconstruction_Loss': recons_loss.detach(),
            'KLD': kld_loss.detach(),
            'KL_Weight': kl_weight
        }


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