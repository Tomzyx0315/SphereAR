import torch


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def karras_sigmas(num_scales, sigma_min, sigma_max, rho, device):
    ramp = torch.linspace(0, 1, num_scales, device=device)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho


def get_weightings(weight_schedule, sigmas, sigma_data):
    snrs = sigmas**-2
    if weight_schedule == "snr":
        return snrs
    if weight_schedule == "snr+1":
        return snrs + 1
    if weight_schedule == "karras":
        return snrs + 1.0 / sigma_data**2
    if weight_schedule == "truncated-snr":
        return torch.clamp(snrs, min=1.0)
    if weight_schedule == "uniform":
        return torch.ones_like(sigmas)
    raise ValueError(f"Unknown weight schedule: {weight_schedule}")


class SphereARConsistencyDistiller:
    """Teacher-model consistency distillation for SphereAR latent tokens."""

    def __init__(
        self,
        teacher,
        student_head,
        target_head,
        cfg_scale=1.0,
        cfg_schedule="linear",
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
        sigma_data=1.0,
        weight_schedule="uniform",
        loss_norm="l2",
        normalize_denoised=True,
        loss_eps=1e-3,
    ):
        self.teacher = teacher
        self.student_head = student_head
        self.target_head = target_head
        self.cfg_scale = cfg_scale
        self.cfg_schedule = cfg_schedule
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.weight_schedule = weight_schedule
        self.loss_norm = loss_norm
        self.normalize_denoised = normalize_denoised
        self.loss_eps = loss_eps

    @property
    def device(self):
        return next(self.teacher.parameters()).device

    @property
    def cfg_mult(self):
        return 2 if self.cfg_scale > 1.0 else 1

    def _cfg_at_pos(self, pos):
        if self.cfg_scale <= 1.0:
            return 1.0
        if self.cfg_schedule == "constant":
            return self.cfg_scale
        if self.cfg_schedule == "linear":
            seq_len = self.teacher.h * self.teacher.w
            return 1.0 + (self.cfg_scale - 1.0) * pos / seq_len
        raise ValueError(f"Unknown cfg schedule: {self.cfg_schedule}")

    def _combine_cfg(self, tensor, pos):
        if self.cfg_mult == 1:
            return tensor
        cond, uncond = torch.chunk(tensor, 2, dim=0)
        cfg = self._cfg_at_pos(pos)
        return uncond + cfg * (cond - uncond)

    def _normalize_tokens(self, tokens):
        if not self.normalize_denoised:
            return tokens
        return self.teacher.vae.normalize(tokens.view(tokens.shape[0], 1, -1)).view(
            tokens.shape[0], -1
        )

    def _run_ar_step(self, ar_input, start_pos, end_pos):
        with torch.no_grad():
            return self.teacher.forward_model(ar_input, start_pos, end_pos)

    @torch.no_grad()
    def encode_real_latents(self, images):
        latents, _ = self.teacher.vae.encode(images)
        return latents

    @torch.no_grad()
    def teacher_forced_conditions(self, class_id, clean_latents):
        teacher = self.teacher
        if self.cfg_mult == 2:
            cond_null = torch.ones_like(class_id) * teacher.num_classes
            class_for_ar = torch.cat([class_id, cond_null], dim=0)
            clean_latents = torch.cat([clean_latents, clean_latents], dim=0)
        else:
            class_for_ar = class_id

        bsz = class_for_ar.shape[0]
        class_tokens = teacher.cls_embedding(class_for_ar).view(
            bsz, teacher.cls_token_num, -1
        )
        prefix_tokens = teacher.proj_in(clean_latents[:, :-1, :])
        x = torch.cat([class_tokens, prefix_tokens], dim=1)

        x = teacher.emb_norm(x)
        for layer in teacher.layers:
            x = layer(x, teacher.freqs_cis)
        x = x[:, -teacher.h * teacher.w :, :]
        x = teacher.norm(x)
        return x + teacher.pos_for_diff.weight

    def build_real_training_batch(self, class_id, real_latents):
        if real_latents is None:
            raise ValueError("real_latents is required for consistency distillation.")
        conds = self.teacher_forced_conditions(class_id, real_latents)
        return real_latents, conds

    def _cond_for_position(self, conds, pos_idx):
        if self.cfg_mult == 1:
            return conds[:, pos_idx, :]
        bsz = conds.shape[0] // self.cfg_mult
        conds = conds.view(self.cfg_mult, bsz, conds.shape[1], conds.shape[2])
        return torch.cat([conds[0, :, pos_idx, :], conds[1, :, pos_idx, :]], dim=0)

    def _denoise_head(self, head, x_sigma, sigma, cond, pos):
        if self.cfg_mult == 2:
            x_in = torch.cat([x_sigma, x_sigma], dim=0)
            sigma_in = torch.cat([sigma, sigma], dim=0)
        else:
            x_in = x_sigma
            sigma_in = sigma
        denoised = head(x_in, sigma_in, cond)
        denoised = self._combine_cfg(denoised, pos)
        return self._normalize_tokens(denoised)

    def _teacher_denoise(self, x_sigma, sigma, cond, pos):
        if self.cfg_mult == 2:
            x_in = torch.cat([x_sigma, x_sigma], dim=0)
            sigma_in = torch.cat([sigma, sigma], dim=0)
        else:
            x_in = x_sigma
            sigma_in = sigma

        flow_t = 1.0 / (1.0 + sigma_in)
        x_flow = x_in * flow_t.view(-1, 1)
        velocity = self.teacher.head.net(x_flow, flow_t.to(torch.float32), cond)
        velocity = self._combine_cfg(velocity, pos)
        base_flow = x_sigma * (1.0 / (1.0 + sigma)).view(-1, 1)
        denoised = base_flow + (1.0 - 1.0 / (1.0 + sigma)).view(-1, 1) * velocity
        return self._normalize_tokens(denoised)

    @torch.no_grad()
    def _heun_teacher_step(self, x_sigma, sigma, next_sigma, cond, pos):
        denoised = self._teacher_denoise(x_sigma, sigma, cond, pos)
        d = (x_sigma - denoised) / sigma.view(-1, 1)
        x_euler = x_sigma + d * (next_sigma - sigma).view(-1, 1)
        denoised_next = self._teacher_denoise(x_euler, next_sigma, cond, pos)
        d_next = (x_euler - denoised_next) / next_sigma.view(-1, 1)
        return x_sigma + 0.5 * (d + d_next) * (next_sigma - sigma).view(-1, 1)

    def _position_loss(self, pred, target, sigmas):
        if self.loss_norm == "l1":
            loss = torch.abs(pred - target).mean(dim=1)
        elif self.loss_norm == "l2":
            loss = ((pred - target) ** 2).mean(dim=1)
        elif self.loss_norm == "pseudo-huber":
            diffs = pred - target
            loss = (
                torch.sqrt(diffs.pow(2).mean(dim=1) + self.loss_eps**2)
                - self.loss_eps
            )
        else:
            raise ValueError(f"Unknown loss norm: {self.loss_norm}")
        weights = get_weightings(self.weight_schedule, sigmas, self.sigma_data)
        return loss * weights

    def consistency_loss(self, clean_latents, conds, num_scales, position_indices=None):
        if num_scales < 2:
            raise ValueError("num_scales must be >= 2 for consistency distillation.")
        student = (
            self.student_head
            if torch.is_grad_enabled()
            else unwrap_model(self.student_head)
        )
        target = unwrap_model(self.target_head)
        sigmas = karras_sigmas(
            num_scales,
            self.sigma_min,
            self.sigma_max,
            self.rho,
            clean_latents.device,
        )
        bsz, seq_len, _latent_dim = clean_latents.shape
        if position_indices is None:
            positions = torch.arange(seq_len, device=clean_latents.device)
        else:
            positions = position_indices.to(clean_latents.device)

        losses = []
        sigma_values = []
        with torch.no_grad():
            target_was_training = target.training
            target.eval()
        try:
            for pos_tensor in positions:
                pos = int(pos_tensor.item())
                x0 = clean_latents[:, pos, :]
                cond = self._cond_for_position(conds, pos)
                indices = torch.randint(
                    0, num_scales - 1, (bsz,), device=clean_latents.device
                )
                sigma = sigmas[indices]
                next_sigma = sigmas[indices + 1]
                noise = torch.randn_like(x0)
                x_sigma = x0 + noise * sigma.view(-1, 1)

                pred = self._denoise_head(student, x_sigma, sigma, cond, pos)
                with torch.no_grad():
                    x_next = self._heun_teacher_step(
                        x_sigma, sigma, next_sigma, cond, pos
                    )
                    target_pred = self._denoise_head(
                        target, x_next, next_sigma, cond, pos
                    )
                losses.append(self._position_loss(pred, target_pred, sigma))
                sigma_values.append(sigma)
        finally:
            if target_was_training:
                target.train()

        all_sigmas = torch.cat(sigma_values, dim=0)
        logs = {
            "cm_sigma": all_sigmas.mean().detach(),
            "cm_sigma_minibatch_max": all_sigmas.max().detach(),
            "cm_num_positions": torch.tensor(
                float(positions.numel()), device=clean_latents.device
            ),
        }
        loss = torch.cat(losses, dim=0).mean()
        return loss, logs

    @torch.no_grad()
    def generate_latents_autoregressive(self, class_id, sampling_sigmas=None):
        teacher = self.teacher
        student = unwrap_model(self.student_head)
        if sampling_sigmas is None:
            sampling_sigmas = [self.sigma_max]
        sigmas = torch.as_tensor(
            sampling_sigmas, device=class_id.device, dtype=torch.float32
        )
        if sigmas.ndim != 1 or sigmas.numel() < 1:
            raise ValueError("sampling_sigmas must be a non-empty 1D sequence.")

        if self.cfg_mult == 2:
            cond_null = torch.ones_like(class_id) * teacher.num_classes
            class_for_ar = torch.cat([class_id, cond_null], dim=0)
        else:
            class_for_ar = class_id

        bsz = class_for_ar.shape[0]
        act_bsz = class_id.shape[0]
        teacher.enable_kv_cache(bsz)
        class_tokens = teacher.cls_embedding(class_for_ar).view(
            bsz, teacher.cls_token_num, -1
        )

        latents = []
        last_pred = None
        for pos in range(teacher.h * teacher.w):
            if pos == 0:
                x = self._run_ar_step(class_tokens, 0, teacher.cls_token_num)
            else:
                ar_token = teacher.proj_in(last_pred)
                x = self._run_ar_step(
                    ar_token,
                    pos + teacher.cls_token_num - 1,
                    pos + teacher.cls_token_num,
                )
            cond = x[:, -1:, :] + teacher.pos_for_diff.weight[pos : pos + 1, :]
            cond = cond.view(-1, cond.shape[-1])
            x_sigma = torch.randn(
                act_bsz,
                teacher.latent_dim,
                device=class_id.device,
                dtype=cond.dtype,
            ) * sigmas[0]
            pred = None
            for sigma_idx, sigma in enumerate(sigmas):
                sigma_batch = torch.full(
                    (act_bsz,),
                    float(sigma.item()),
                    device=class_id.device,
                    dtype=torch.float32,
                )
                pred = self._denoise_head(student, x_sigma, sigma_batch, cond, pos)
                if sigma_idx + 1 < sigmas.numel():
                    x_sigma = pred + torch.randn_like(pred) * sigmas[sigma_idx + 1]
            pred = self._normalize_tokens(pred).view(act_bsz, 1, -1)
            latents.append(pred)
            last_pred = pred if self.cfg_mult == 1 else torch.cat([pred, pred], dim=0)

        return torch.cat(latents, dim=1)

    def decode_latents(self, latents):
        return self.teacher.vae.decode(latents)
