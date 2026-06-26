import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from SphereAR.gan.discriminator_patchgan import NLayerDiscriminator
from SphereAR.gan.discriminator_stylegan import Discriminator as StyleGANDiscriminator


def spectral_norm(module):
    return nn.utils.spectral_norm(module)


class ResNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.skip = (
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            if in_channels != out_channels
            else nn.Identity()
        )
        self.downsample = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        h = F.leaky_relu(x, negative_slope=0.2)
        h = self.conv1(h)
        h = F.leaky_relu(h, negative_slope=0.2)
        h = self.conv2(h)
        h = self.downsample(h)
        return h + self.downsample(self.skip(x))


class ProjectionResNetDiscriminator(nn.Module):
    """Class-conditional spectral-norm ResNet discriminator."""

    requires_labels = True

    def __init__(self, image_size, num_classes, base_channels=64, max_channels=512):
        super().__init__()
        if image_size & (image_size - 1) != 0 or image_size < 16:
            raise ValueError(f"image_size must be a power of two >= 16, got {image_size}")

        self.num_classes = num_classes
        self.from_rgb = spectral_norm(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1)
        )

        blocks = []
        in_channels = base_channels
        num_downsamples = int(math.log2(image_size)) - 2
        for level in range(num_downsamples):
            out_channels = min(base_channels * (2 ** (level + 1)), max_channels)
            blocks.append(ResNetDownBlock(in_channels, out_channels))
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

        self.final_linear = spectral_norm(nn.Linear(in_channels, 1))
        self.class_embed = nn.Embedding(num_classes, in_channels)
        nn.init.normal_(self.class_embed.weight, mean=0.0, std=0.02)

    def forward(self, x, labels):
        if labels is None:
            raise ValueError("ProjectionResNetDiscriminator requires class labels.")
        h = self.from_rgb(x)
        h = self.blocks(h)
        h = F.leaky_relu(h, negative_slope=0.2)
        h = h.sum(dim=(2, 3))
        logits = self.final_linear(h)
        projection = (self.class_embed(labels) * h).sum(dim=1, keepdim=True)
        return logits + projection / math.sqrt(h.shape[1])


class ProjectionLatentGridDiscriminator(nn.Module):
    """Class-conditional discriminator over full latent grids."""

    requires_labels = True

    def __init__(self, latent_dim, grid_size, num_classes, base_channels=64):
        super().__init__()
        self.grid_size = grid_size
        self.net = ProjectionResNetDiscriminator(
            image_size=grid_size,
            num_classes=num_classes,
            base_channels=base_channels,
        )
        self.net.from_rgb = spectral_norm(
            nn.Conv2d(latent_dim, base_channels, kernel_size=3, padding=1)
        )

    def forward(self, latents, labels):
        bsz, seq_len, channels = latents.shape
        if seq_len != self.grid_size * self.grid_size:
            raise ValueError(
                f"Expected {self.grid_size * self.grid_size} latent tokens, got {seq_len}"
            )
        x = latents.view(bsz, self.grid_size, self.grid_size, channels).permute(
            0, 3, 1, 2
        )
        return self.net(x, labels)


class ProjectionLatentTokenDiscriminator(nn.Module):
    """Class- and position-conditional discriminator over individual tokens."""

    requires_labels = True
    requires_positions = True

    def __init__(self, latent_dim, num_positions, num_classes, hidden_dim=256):
        super().__init__()
        self.num_positions = num_positions
        self.input_proj = spectral_norm(nn.Linear(latent_dim, hidden_dim))
        self.hidden = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_linear = spectral_norm(nn.Linear(hidden_dim, 1))
        self.class_embed = nn.Embedding(num_classes, hidden_dim)
        self.position_embed = nn.Embedding(num_positions, hidden_dim)
        nn.init.normal_(self.class_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embed.weight, mean=0.0, std=0.02)

    def forward(self, tokens, labels, positions):
        if labels is None or positions is None:
            raise ValueError("Latent-token discriminator requires labels and positions.")
        h = self.input_proj(tokens)
        h = self.hidden(h)
        logits = self.final_linear(h)
        cond = self.class_embed(labels) + self.position_embed(positions)
        projection = (cond * h).sum(dim=1, keepdim=True)
        return logits + projection / math.sqrt(h.shape[1])


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def discriminator_forward(discriminator, inputs, labels=None, positions=None):
    module = unwrap_model(discriminator)
    if getattr(module, "requires_positions", False):
        return discriminator(inputs, labels, positions)
    if getattr(module, "requires_labels", False):
        return discriminator(inputs, labels)
    return discriminator(inputs)


def build_discriminator(
    gan_domain,
    disc_type,
    image_size,
    patch_size,
    latent_dim,
    num_classes,
    disc_dim=64,
):
    if gan_domain == "none":
        return None
    if gan_domain == "latent_grid":
        return ProjectionLatentGridDiscriminator(
            latent_dim=latent_dim,
            grid_size=image_size // patch_size,
            num_classes=num_classes,
            base_channels=disc_dim,
        )
    if gan_domain == "latent_token":
        return ProjectionLatentTokenDiscriminator(
            latent_dim=latent_dim,
            num_positions=(image_size // patch_size) ** 2,
            num_classes=num_classes,
            hidden_dim=max(128, disc_dim * 4),
        )
    if gan_domain != "image":
        raise ValueError(f"Unknown GAN domain: {gan_domain}")
    if disc_type == "resnet":
        return ProjectionResNetDiscriminator(
            image_size=image_size,
            num_classes=num_classes,
            base_channels=disc_dim,
        )
    if disc_type == "patchgan":
        return NLayerDiscriminator(input_nc=3, n_layers=3, ndf=disc_dim)
    if disc_type == "stylegan":
        return StyleGANDiscriminator(input_nc=3, image_size=image_size)
    raise ValueError(f"Unknown discriminator type: {disc_type}")


def discriminator_loss(logits_real, logits_fake, loss_type):
    if loss_type == "hinge":
        loss_real = torch.mean(F.relu(1.0 - logits_real))
        loss_fake = torch.mean(F.relu(1.0 + logits_fake))
        return 0.5 * (loss_real + loss_fake)
    if loss_type == "non-saturating":
        loss_real = F.binary_cross_entropy_with_logits(
            logits_real, torch.ones_like(logits_real)
        )
        loss_fake = F.binary_cross_entropy_with_logits(
            logits_fake, torch.zeros_like(logits_fake)
        )
        return 0.5 * (loss_real + loss_fake)
    raise ValueError(f"Unknown discriminator loss: {loss_type}")


def generator_loss(logits_fake, loss_type):
    if loss_type == "hinge":
        return -torch.mean(logits_fake)
    if loss_type == "non-saturating":
        return F.binary_cross_entropy_with_logits(
            logits_fake, torch.ones_like(logits_fake)
        )
    raise ValueError(f"Unknown generator loss: {loss_type}")
