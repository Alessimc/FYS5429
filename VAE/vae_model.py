import torch
from torch import nn
from torch.nn import functional as F
import math

class VAE(nn.Module):
    def __init__(self, in_channels=2, latent_dim=128, hidden_dims=None):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # 416 -> 13x13 after 5 downsamples (stride=2)
        self.final_feature_size = 13
        self.encoder_output_dim = hidden_dims[-1] * self.final_feature_size * self.final_feature_size

        self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_var = nn.Linear(self.encoder_output_dim, latent_dim)

        # Build Decoder
        self.decoder_input = nn.Linear(latent_dim, self.encoder_output_dim)
        hidden_dims.reverse()

        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=2,  # 2-channel output
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, self.final_feature_size, self.final_feature_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), input, mu, log_var

    def loss_function(self, *args, M_N=1.0):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + M_N * kld_loss

        return {
            'loss': loss,
            'Reconstruction_Loss': recons_loss.detach(),
            'KLD': kld_loss.detach()
        }

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)

    def generate(self, x):
        return self.forward(x)[0]


class REDNetVAE(nn.Module):
    def __init__(self, num_layers=5, num_features=256, in_channels=2, out_channels=2, latent_dim=128):
        super(REDNetVAE, self).__init__()
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


