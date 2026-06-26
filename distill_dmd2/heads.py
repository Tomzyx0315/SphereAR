import torch
import torch.nn as nn

from SphereAR.diff_head import MlpEncoder


class OneStepHead(nn.Module):
    """One-step latent generator initialized from a trained SphereAR diffusion head."""

    def __init__(self, teacher_head):
        super().__init__()
        teacher_net = teacher_head.net
        self.ch_target = teacher_head.ch_target
        self.net = MlpEncoder(
            in_channels=teacher_net.in_channels,
            model_channels=teacher_net.model_channels,
            z_channels=teacher_net.cond_embed.in_features,
            num_res_blocks=teacher_net.num_res_blocks,
            num_ada_ln_blocks=len(teacher_net.ada_ln_blocks),
            grad_checkpointing=teacher_net.grad_checkpointing,
        )
        self.net.load_state_dict(teacher_net.state_dict(), strict=True)

    def forward(self, noise, cond):
        t = torch.zeros(noise.shape[0], device=noise.device, dtype=torch.float32)
        velocity = self.net(noise, t, cond)
        return noise + velocity


class FakeScoreHead(nn.Module):
    """Trainable fake score model used by the distribution matching loss."""

    def __init__(self, teacher_head):
        super().__init__()
        teacher_net = teacher_head.net
        self.ch_target = teacher_head.ch_target
        self.net = MlpEncoder(
            in_channels=teacher_net.in_channels,
            model_channels=teacher_net.model_channels,
            z_channels=teacher_net.cond_embed.in_features,
            num_res_blocks=teacher_net.num_res_blocks,
            num_ada_ln_blocks=len(teacher_net.ada_ln_blocks),
            grad_checkpointing=teacher_net.grad_checkpointing,
        )
        self.net.load_state_dict(teacher_net.state_dict(), strict=True)

    def forward(self, x, t, cond):
        return self.net(x, t, cond)

