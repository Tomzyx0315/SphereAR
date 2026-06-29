import torch
import torch.nn as nn

from SphereAR.diff_head import MlpEncoder


class ConsistencyHead(nn.Module):
    """Consistency denoiser initialized from a trained SphereAR diffusion head."""

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

    @staticmethod
    def sigma_to_flow_t(sigma):
        return 1.0 / (1.0 + sigma)

    def forward(self, x_sigma, sigma, cond):
        """Predict clean latent tokens from additive-noise latents.

        SphereAR's diffusion head was trained with x_t = (1 - t) noise + t x0.
        Consistency models use x_sigma = x0 + sigma noise.  The mapping
        t = 1 / (1 + sigma) gives x_t = t * x_sigma, so the pretrained
        velocity head can be reused directly.
        """
        flow_t = self.sigma_to_flow_t(sigma).to(dtype=torch.float32)
        x_flow = x_sigma * flow_t.view(-1, 1)
        velocity = self.net(x_flow, flow_t, cond)
        return x_flow + (1.0 - flow_t).view(-1, 1) * velocity

