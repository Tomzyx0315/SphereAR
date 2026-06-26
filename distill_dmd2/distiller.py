from contextlib import nullcontext

import torch
import torch.nn.functional as F


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


class SphereARDMD2Distiller:
    """Wraps a frozen SphereAR teacher with one-step student and fake score heads."""

    def __init__(
        self,
        teacher,
        student_head,
        fake_score_head,
        discriminator,
        cfg_scale=1.0,
        cfg_schedule="linear",
        gan_loss_type="hinge",
        dm_eps=1e-6,
    ):
        self.teacher = teacher
        self.student_head = student_head
        self.fake_score_head = fake_score_head
        self.discriminator = discriminator
        self.cfg_scale = cfg_scale
        self.cfg_schedule = cfg_schedule
        self.gan_loss_type = gan_loss_type
        self.dm_eps = dm_eps

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

    def _student_step(self, noise, cond, pos):
        cfg_mult = self.cfg_mult
        noise_for_head = noise if cfg_mult == 1 else torch.cat([noise, noise], dim=0)
        student_head = (
            self.student_head if torch.is_grad_enabled() else unwrap_model(self.student_head)
        )
        pred = student_head(noise_for_head, cond)
        pred = self._combine_cfg(pred, pos)
        pred = self.teacher.vae.normalize(pred.view(pred.shape[0], 1, -1))
        return pred

    def _run_ar_step(self, ar_input, start_pos, end_pos):
        with torch.no_grad():
            return self.teacher.forward_model(ar_input, start_pos, end_pos)

    @torch.no_grad()
    def encode_real_latents(self, images):
        latents, _ = self.teacher.vae.encode(images)
        return latents

    @torch.no_grad()
    def sample_teacher_latents(
        self,
        class_id,
        sample_steps,
        cfg_scale=1.0,
        cfg_schedule="linear",
    ):
        teacher = self.teacher
        if cfg_scale > 1.0:
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

        last_pred = None
        latents = []
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
            last_pred = teacher.head_sample(
                x[:, -1:, :],
                pos,
                sample_steps,
                cfg_scale,
                cfg_schedule,
            )
            latents.append(last_pred)

        return torch.cat(latents, dim=1)[:act_bsz]

    @torch.no_grad()
    def teacher_forced_conditions(self, class_id, teacher_latents):
        teacher = self.teacher
        cfg_mult = self.cfg_mult
        if cfg_mult == 2:
            cond_null = torch.ones_like(class_id) * teacher.num_classes
            class_for_ar = torch.cat([class_id, cond_null], dim=0)
            prefix_latents = torch.cat([teacher_latents, teacher_latents], dim=0)
        else:
            class_for_ar = class_id
            prefix_latents = teacher_latents

        bsz = class_for_ar.shape[0]
        class_tokens = teacher.cls_embedding(class_for_ar).view(
            bsz, teacher.cls_token_num, -1
        )
        prefix_tokens = teacher.proj_in(prefix_latents[:, :-1, :])
        x = torch.cat([class_tokens, prefix_tokens], dim=1)

        x = teacher.emb_norm(x)
        for layer in teacher.layers:
            x = layer(x, teacher.freqs_cis)
        x = x[:, -teacher.h * teacher.w :, :]
        x = teacher.norm(x)
        return x + teacher.pos_for_diff.weight

    def _generate_from_conditions(self, class_id, conds, requires_grad):
        bsz = class_id.shape[0]
        seq_len = self.teacher.h * self.teacher.w
        latents = []
        head_context = nullcontext() if requires_grad else torch.no_grad()
        with head_context:
            for pos in range(seq_len):
                cond = conds[:, pos, :]
                noise = torch.randn(
                    bsz, self.teacher.latent_dim, device=class_id.device
                )
                latents.append(self._student_step(noise, cond, pos))
        return torch.cat(latents, dim=1)

    def generate_latents_teacher_forcing(
        self,
        class_id,
        requires_grad,
        teacher_sample_steps,
        teacher_cfg_scale=1.0,
        teacher_cfg_schedule="linear",
    ):
        teacher_latents = self.sample_teacher_latents(
            class_id,
            sample_steps=teacher_sample_steps,
            cfg_scale=teacher_cfg_scale,
            cfg_schedule=teacher_cfg_schedule,
        )
        conds = self.teacher_forced_conditions(class_id, teacher_latents)
        latents = self._generate_from_conditions(class_id, conds, requires_grad)
        return latents, conds

    def generate_latents_real_prefix(self, class_id, real_latents, requires_grad):
        if real_latents is None:
            raise ValueError("real prefix mode requires real_latents.")
        conds = self.teacher_forced_conditions(class_id, real_latents)
        latents = self._generate_from_conditions(class_id, conds, requires_grad)
        return latents, conds

    def generate_latents_autoregressive(self, class_id, requires_grad=False):
        """Generate a full latent grid by feeding student tokens back to the AR trunk."""
        if requires_grad:
            raise ValueError("Autoregressive student-prefix generation is sampling-only.")
        teacher = self.teacher
        cfg_mult = self.cfg_mult
        if cfg_mult == 2:
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
        conds = []
        last_pred = None
        with torch.no_grad():
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
                cond_flat = cond.view(-1, cond.shape[-1])
                conds.append(cond_flat)

                noise = torch.randn(
                    act_bsz, teacher.latent_dim, device=class_id.device
                )
                pred = self._student_step(noise, cond_flat, pos)
                latents.append(pred)
                last_pred = pred if cfg_mult == 1 else torch.cat([pred, pred], dim=0)

        return torch.cat(latents, dim=1), torch.stack(conds, dim=1)

    def generate_latents(
        self,
        class_id,
        requires_grad,
        prefix_mode="teacher_forcing",
        real_latents=None,
        teacher_sample_steps=100,
        teacher_cfg_scale=1.0,
        teacher_cfg_schedule="linear",
    ):
        if prefix_mode == "teacher_forcing":
            return self.generate_latents_teacher_forcing(
                class_id,
                requires_grad=requires_grad,
                teacher_sample_steps=teacher_sample_steps,
                teacher_cfg_scale=teacher_cfg_scale,
                teacher_cfg_schedule=teacher_cfg_schedule,
            )
        if prefix_mode == "real":
            return self.generate_latents_real_prefix(
                class_id,
                real_latents=real_latents,
                requires_grad=requires_grad,
            )
        raise ValueError(f"Unknown prefix mode: {prefix_mode}")

    def decode_latents(self, latents):
        return self.teacher.vae.decode(latents)

    def _score_x0_prediction(self, score_head, x_t, t, cond, pos):
        cfg_mult = self.cfg_mult
        if cfg_mult == 2:
            x_t_in = torch.cat([x_t, x_t], dim=0)
            t_in = torch.cat([t, t], dim=0)
        else:
            x_t_in = x_t
            t_in = t
        velocity = score_head(x_t_in, t_in, cond)
        velocity = self._combine_cfg(velocity, pos)
        return x_t + (1.0 - t.view(-1, 1)) * velocity

    def _select_positions(self, generated_latents, conds, position_indices):
        if position_indices is None:
            seq_len = generated_latents.shape[1]
            positions = torch.arange(seq_len, device=generated_latents.device)
            return generated_latents, conds, positions
        positions = position_indices.to(generated_latents.device)
        return generated_latents[:, positions, :], conds[:, positions, :], positions

    def distribution_matching_loss(self, generated_latents, conds, position_indices=None):
        generated_latents, conds, positions = self._select_positions(
            generated_latents, conds, position_indices
        )
        bsz, num_pos, latent_dim = generated_latents.shape
        x = generated_latents.permute(1, 0, 2).reshape(num_pos * bsz, latent_dim)
        x_detached = x.detach()

        with torch.no_grad():
            t = torch.randn(x.shape[0], device=x.device).sigmoid()
            noise = torch.randn_like(x_detached)
            x_t = (1.0 - t.view(-1, 1)) * noise + t.view(-1, 1) * x_detached

            teacher_head = self.teacher.head
            fake_head = unwrap_model(self.fake_score_head)
            cond_by_branch = None
            cond_token_major = None
            if self.cfg_mult == 2:
                cond_by_branch = conds.reshape(self.cfg_mult, bsz, num_pos, -1)
            else:
                cond_token_major = conds.permute(1, 0, 2).reshape(num_pos * bsz, -1)

            real_x0 = []
            fake_x0 = []
            for pos_idx, pos_tensor in enumerate(positions):
                pos = int(pos_tensor.item())
                token_slice = slice(pos_idx * bsz, (pos_idx + 1) * bsz)
                if self.cfg_mult == 2:
                    cond_pos = torch.cat(
                        [
                            cond_by_branch[0, :, pos_idx, :],
                            cond_by_branch[1, :, pos_idx, :],
                        ],
                        dim=0,
                    )
                else:
                    cond_pos = cond_token_major[token_slice]
                real_x0.append(
                    self._score_x0_prediction(
                        teacher_head.net,
                        x_t[token_slice],
                        t[token_slice],
                        cond_pos,
                        pos,
                    )
                )
                fake_x0.append(
                    self._score_x0_prediction(
                        fake_head,
                        x_t[token_slice],
                        t[token_slice],
                        cond_pos,
                        pos,
                    )
                )

            real_x0 = torch.cat(real_x0, dim=0)
            fake_x0 = torch.cat(fake_x0, dim=0)
            p_real = x_detached - real_x0
            p_fake = x_detached - fake_x0
            weight = p_real.abs().mean(dim=1, keepdim=True).clamp_min(self.dm_eps)
            grad = torch.nan_to_num((p_real - p_fake) / weight)

        loss = 0.5 * F.mse_loss(x, (x_detached - grad).detach(), reduction="mean")
        return loss, {
            "dm_grad_norm": grad.norm().detach(),
            "dm_real_fake_gap": (real_x0 - fake_x0).abs().mean().detach(),
        }

    def fake_score_loss(self, generated_latents, conds, position_indices=None):
        generated_latents, conds, _positions = self._select_positions(
            generated_latents, conds, position_indices
        )
        bsz, num_pos, latent_dim = generated_latents.shape
        x = (
            generated_latents.detach()
            .permute(1, 0, 2)
            .reshape(num_pos * bsz, latent_dim)
        )
        t = torch.randn(x.shape[0], device=x.device).sigmoid()
        noise = torch.randn_like(x)
        x_t = (1.0 - t.view(-1, 1)) * noise + t.view(-1, 1) * x
        target_velocity = x - noise

        if self.cfg_mult == 2:
            cond_by_branch = conds.reshape(self.cfg_mult, bsz, num_pos, -1)
            cond = torch.cat(
                [
                    cond_by_branch[0].permute(1, 0, 2).reshape(num_pos * bsz, -1),
                    cond_by_branch[1].permute(1, 0, 2).reshape(num_pos * bsz, -1),
                ],
                dim=0,
            )
            x_t = torch.cat([x_t, x_t], dim=0)
            t = torch.cat([t, t], dim=0)
            target_velocity = torch.cat([target_velocity, target_velocity], dim=0)
        else:
            cond = conds.permute(1, 0, 2).reshape(num_pos * bsz, -1)

        pred_velocity = self.fake_score_head(x_t, t, cond)
        return F.mse_loss(pred_velocity, target_velocity, reduction="mean")
