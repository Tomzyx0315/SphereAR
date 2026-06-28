import argparse
import math
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch


def _identity_compile(fn=None, *args, **kwargs):
    if fn is None:
        return lambda inner_fn: inner_fn
    return fn


if "--compile-model" not in sys.argv:
    torch.compile = _identity_compile

import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from SphereAR.dataset import build_dataset
from SphereAR.model import create_model, get_model_args
from SphereAR.utils import create_logger, requires_grad
from distill_dmd2.ar_utils import (
    all_reduce_trainable_grads,
    ar_backbone_parameters,
    ar_backbone_state_dict,
    load_ar_backbone_state_dict,
    set_ar_backbone_trainable,
    set_module_requires_grad,
)
from distill_dmd2.distiller import SphereARDMD2Distiller, unwrap_model
from distill_dmd2.gan import (
    build_discriminator,
    discriminator_forward,
    discriminator_loss,
    generator_loss,
)
from distill_dmd2.heads import FakeScoreHead, OneStepHead


def init_distributed_mode(args):
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.gpu = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", args.gpu)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank,
        device_id=device,
    )
    dist.barrier()
    return device


def create_optimizer(params, lr, weight_decay, betas=(0.9, 0.95)):
    return torch.optim.AdamW(
        [p for p in params if p.requires_grad],
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )


def load_teacher(args, device, logger):
    teacher = create_model(args, device)
    checkpoint = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    if "ema" in checkpoint and not args.teacher_no_ema:
        state = checkpoint["ema"]
        logger.info("Loaded teacher EMA weights.")
    elif "model" in checkpoint:
        state = checkpoint["model"]
        logger.info("Loaded teacher model weights.")
    else:
        raise ValueError(f"Cannot find model weights in {args.teacher_ckpt}")
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    requires_grad(teacher, False)
    return teacher


def save_checkpoint(
    args,
    epoch,
    step,
    teacher,
    student_head,
    fake_score_head,
    discriminator,
    opt_student,
    opt_fake,
    opt_disc,
    logger,
    keep_epoch=False,
):
    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint = {
        "student_head": unwrap_model(student_head).state_dict(),
        "fake_score_head": unwrap_model(fake_score_head).state_dict(),
        "optimizer_student": opt_student.state_dict(),
        "optimizer_fake": opt_fake.state_dict(),
        "epoch": epoch,
        "step": step,
        "self_forcing": args.self_forcing,
        "args": args,
    }
    if args.self_forcing:
        checkpoint["ar_backbone"] = ar_backbone_state_dict(teacher)
    if discriminator is not None:
        checkpoint["discriminator"] = unwrap_model(discriminator).state_dict()
    if opt_disc is not None:
        checkpoint["optimizer_disc"] = opt_disc.state_dict()
    path = os.path.join(args.results_dir, "last.pt")
    torch.save(checkpoint, path)
    if keep_epoch and args.keep_freq > 0 and epoch > 0 and epoch % args.keep_freq == 0:
        torch.save(checkpoint, os.path.join(args.results_dir, f"epoch_{epoch}.pt"))
    logger.info(f"Saved distillation checkpoint to {path}")


def save_image_grid(images, path):
    count = len(images)
    if count == 0:
        return
    height, width = images[0].shape[:2]
    cols = math.ceil(count**0.5)
    rows = math.ceil(count / cols)
    grid = Image.new("RGB", (cols * width, rows * height))
    for idx, image in enumerate(images):
        row, col = divmod(idx, cols)
        grid.paste(Image.fromarray(image), (col * width, row * height))
    grid.save(path)


@torch.no_grad()
def save_preview_samples(args, distiller, step, device, ptdtype, logger):
    if args.preview_every <= 0 or args.preview_num <= 0:
        return

    preview_dir = args.preview_dir or os.path.join(args.results_dir, "previews")
    step_dir = os.path.join(preview_dir, f"step_{step:07d}")
    os.makedirs(step_dir, exist_ok=True)

    student = unwrap_model(distiller.student_head)
    teacher_was_training = distiller.teacher.training
    student_was_training = student.training
    distiller.teacher.eval()
    student.eval()

    saved_images = []
    preview_batch_size = args.preview_batch_size
    if preview_batch_size <= 0:
        preview_batch_size = args.preview_num

    try:
        with torch.random.fork_rng(devices=[args.gpu]):
            torch.manual_seed(args.preview_seed + step)
            image_idx = 0
            remaining = args.preview_num
            while remaining > 0:
                current_batch = min(preview_batch_size, remaining)
                class_id = torch.randint(
                    0, args.num_classes, (current_batch,), device=device
                )
                with torch.amp.autocast("cuda", dtype=ptdtype):
                    latents, _ = distiller.generate_latents_autoregressive(class_id)
                    samples = distiller.decode_latents(latents)
                samples = (
                    torch.clamp(127.5 * samples + 128.0, 0, 255)
                    .permute(0, 2, 3, 1)
                    .to("cpu", dtype=torch.uint8)
                    .numpy()
                )
                for sample in samples:
                    Image.fromarray(sample).save(
                        os.path.join(step_dir, f"{image_idx:03d}.png")
                    )
                    saved_images.append(sample)
                    image_idx += 1
                remaining -= current_batch
        save_image_grid(saved_images, os.path.join(step_dir, "grid.png"))
        logger.info(f"Saved {len(saved_images)} preview samples to {step_dir}")
    finally:
        if args.self_forcing:
            set_ar_backbone_trainable(distiller.teacher, True)
        else:
            distiller.teacher.train(teacher_was_training)
        student.train(student_was_training)


def checkpoint_self_forcing(checkpoint):
    if "self_forcing" in checkpoint:
        return bool(checkpoint["self_forcing"])
    ckpt_args = checkpoint.get("args", None)
    if ckpt_args is not None and hasattr(ckpt_args, "self_forcing"):
        return bool(ckpt_args.self_forcing)
    return "ar_backbone" in checkpoint


def load_distill_weights(
    args,
    checkpoint,
    teacher,
    student_head,
    fake_score_head,
    discriminator,
    logger,
    source,
    require_stage_match,
):
    ckpt_is_self_forcing = checkpoint_self_forcing(checkpoint)
    if require_stage_match and ckpt_is_self_forcing != args.self_forcing:
        current_stage = "self-forcing" if args.self_forcing else "teacher-forcing"
        checkpoint_stage = "self-forcing" if ckpt_is_self_forcing else "teacher-forcing"
        raise ValueError(
            f"--resume expects a {current_stage} checkpoint, but {source} is a "
            f"{checkpoint_stage} checkpoint. Use --init-from for weight-only "
            "initialization between stages."
        )

    unwrap_model(student_head).load_state_dict(checkpoint["student_head"], strict=True)
    unwrap_model(fake_score_head).load_state_dict(
        checkpoint["fake_score_head"], strict=True
    )
    if discriminator is not None and "discriminator" in checkpoint:
        unwrap_model(discriminator).load_state_dict(
            checkpoint["discriminator"], strict=True
        )
    elif require_stage_match and discriminator is not None:
        raise ValueError(
            f"--resume expected discriminator weights in {source}. Check that "
            "--gan-domain matches the resumed run."
        )
    elif require_stage_match and discriminator is None and "discriminator" in checkpoint:
        raise ValueError(
            f"--resume found discriminator weights in {source}, but the current "
            "configuration has --gan-domain none."
        )
    if "ar_backbone" in checkpoint:
        if args.self_forcing:
            load_ar_backbone_state_dict(teacher, checkpoint["ar_backbone"])
            logger.info(f"Loaded self-forcing AR backbone weights from {source}.")
        else:
            logger.info(
                f"Ignored AR backbone weights in {source} because --self-forcing "
                "is disabled."
            )
    elif args.self_forcing:
        if require_stage_match:
            raise ValueError(
                f"{source} is a self-forcing resume checkpoint but has no "
                "ar_backbone state."
            )
        logger.info(
            f"{source} has no ar_backbone; self-forcing keeps the AR backbone "
            "loaded from --teacher-ckpt."
        )


def load_init_from(args, teacher, student_head, fake_score_head, discriminator, logger):
    if not args.init_from:
        return
    if not os.path.exists(args.init_from):
        raise FileNotFoundError(f"--init-from checkpoint not found: {args.init_from}")
    checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=False)
    load_distill_weights(
        args,
        checkpoint,
        teacher,
        student_head,
        fake_score_head,
        discriminator,
        logger,
        args.init_from,
        require_stage_match=False,
    )
    logger.info(
        f"Initialized distillation weights from {args.init_from}; optimizer, epoch, "
        "and step were not loaded."
    )


def load_resume(args, teacher, student_head, fake_score_head, discriminator,
                opt_student, opt_fake, opt_disc, logger):
    if not args.resume:
        return 0, 0
    if not os.path.exists(args.resume):
        raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    load_distill_weights(
        args,
        checkpoint,
        teacher,
        student_head,
        fake_score_head,
        discriminator,
        logger,
        args.resume,
        require_stage_match=True,
    )
    for key in ("optimizer_student", "optimizer_fake", "epoch", "step"):
        if key not in checkpoint:
            raise KeyError(f"--resume checkpoint {args.resume} is missing {key}.")
    opt_student.load_state_dict(checkpoint["optimizer_student"])
    opt_fake.load_state_dict(checkpoint["optimizer_fake"])
    if opt_disc is not None:
        if "optimizer_disc" not in checkpoint:
            raise KeyError(
                f"--resume checkpoint {args.resume} is missing optimizer_disc."
            )
        opt_disc.load_state_dict(checkpoint["optimizer_disc"])
    elif "optimizer_disc" in checkpoint:
        raise ValueError(
            f"--resume found optimizer_disc in {args.resume}, but the current "
            "configuration has --gan-domain none."
        )
    logger.info(f"Resumed distillation from {args.resume}")
    return checkpoint["epoch"], checkpoint["step"]


def reduce_scalar(value, device):
    if not torch.is_tensor(value):
        value = torch.tensor(value, device=device, dtype=torch.float32)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value / dist.get_world_size()


def ddp_sync_context(modules, sync_gradients):
    stack = ExitStack()
    if not sync_gradients:
        for module in modules:
            if module is not None and hasattr(module, "no_sync"):
                stack.enter_context(module.no_sync())
    return stack


def discriminator_for_generator(discriminator, fake_inputs, fake_classes, positions=None):
    disc = unwrap_model(discriminator)
    was_training = disc.training
    set_module_requires_grad(disc, False)
    disc.eval()
    logits_fake = discriminator_forward(disc, fake_inputs, fake_classes, positions)
    if was_training:
        disc.train()
    set_module_requires_grad(disc, True)
    return logits_fake


def sample_position_indices(seq_len, sample_size, device):
    if sample_size <= 0 or sample_size >= seq_len:
        return None
    return torch.randperm(seq_len, device=device)[:sample_size].sort().values


def token_gan_inputs(latents, classes, position_indices):
    bsz, seq_len, latent_dim = latents.shape
    if position_indices is None:
        positions = torch.arange(seq_len, device=latents.device)
    else:
        positions = position_indices.to(latents.device)
    tokens = latents[:, positions, :].reshape(-1, latent_dim)
    labels = classes[:, None].expand(-1, positions.shape[0]).reshape(-1)
    flat_positions = positions[None, :].expand(bsz, -1).reshape(-1)
    return tokens, labels, flat_positions


def gan_inputs(args, distiller, images, latents, classes, position_indices):
    if args.gan_domain == "none":
        return None, None, None
    if args.gan_domain == "image":
        inputs = images if images is not None else distiller.decode_latents(latents)
        return inputs, classes, None
    if args.gan_domain == "latent_grid":
        return latents, classes, None
    if args.gan_domain == "latent_token":
        return token_gan_inputs(latents, classes, position_indices)
    raise ValueError(f"Unknown GAN domain: {args.gan_domain}")


def count_parameters(params):
    return sum(param.numel() for param in params if param.requires_grad)


def main(args):
    assert torch.cuda.is_available(), "DMD2 distillation requires CUDA."
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = init_distributed_mode(args)
    rank = dist.get_rank()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        logger = create_logger(args.results_dir)
        logger.info(f"Experiment directory created at {args.results_dir}")
    else:
        logger = create_logger(None)
    logger.info(args)
    logger.info(f"torch.compile enabled: {args.compile_model}")

    if args.resume and args.init_from:
        raise ValueError("--resume and --init-from are mutually exclusive.")
    if args.self_forcing and args.prefix_mode != "teacher_forcing":
        raise ValueError("--self-forcing is only supported with --prefix-mode teacher_forcing.")
    teacher = load_teacher(args, device, logger)
    set_ar_backbone_trainable(teacher, args.self_forcing)
    student_head = OneStepHead(teacher.head).to(device)
    fake_score_head = FakeScoreHead(teacher.head).to(device)
    discriminator = build_discriminator(
        gan_domain=args.gan_domain,
        disc_type=args.disc_type,
        image_size=args.image_size,
        patch_size=args.patch_size,
        latent_dim=args.latent_dim,
        num_classes=args.num_classes,
        disc_dim=args.disc_dim,
    )
    if discriminator is not None:
        discriminator = discriminator.to(device, memory_format=torch.channels_last)

    student_head = DDP(student_head, device_ids=[args.gpu])
    fake_score_head = DDP(fake_score_head, device_ids=[args.gpu])
    if discriminator is not None:
        discriminator = DDP(discriminator, device_ids=[args.gpu])

    generator_params = list(student_head.parameters())
    if args.self_forcing:
        generator_params += list(ar_backbone_parameters(teacher))
    opt_student = create_optimizer(generator_params, args.student_lr, args.weight_decay)
    opt_fake = create_optimizer(
        fake_score_head.parameters(), args.fake_score_lr, args.weight_decay
    )
    opt_disc = (
        create_optimizer(discriminator.parameters(), args.disc_lr, args.weight_decay)
        if discriminator is not None
        else None
    )

    load_init_from(
        args,
        teacher,
        student_head,
        fake_score_head,
        discriminator,
        logger,
    )
    start_epoch, train_steps = load_resume(
        args,
        teacher,
        student_head,
        fake_score_head,
        discriminator,
        opt_student,
        opt_fake,
        opt_disc,
        logger,
    )

    dataset = build_dataset(args)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    per_gpu_batch = args.global_batch_size // (
        dist.get_world_size() * args.grad_accum_steps
    )
    if per_gpu_batch <= 0:
        raise ValueError(
            "global_batch_size must be at least world_size * grad_accum_steps."
        )
    loader = DataLoader(
        dataset,
        batch_size=per_gpu_batch,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    logger.info(
        f"Effective global batch size: {args.global_batch_size}; "
        f"per-GPU microbatch: {per_gpu_batch}; "
        f"gradient accumulation steps: {args.grad_accum_steps}"
    )
    logger.info(
        f"Self-forcing: {args.self_forcing}; "
        f"trainable student/generator params: {count_parameters(generator_params):,}; "
        f"trainable AR backbone params: {count_parameters(list(ar_backbone_parameters(teacher))):,}"
    )

    distiller = SphereARDMD2Distiller(
        teacher=teacher,
        student_head=student_head,
        fake_score_head=fake_score_head,
        discriminator=discriminator,
        cfg_scale=args.cfg_scale,
        cfg_schedule=args.cfg_schedule,
        gan_loss_type=args.gan_loss,
    )

    ptdtype = {"none": torch.float32, "bf16": torch.bfloat16}[args.mixed_precision]
    steps_per_epoch = int(len(dataset) / args.global_batch_size)
    max_steps = args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch

    running = {}
    running_counts = {}
    log_count = 0
    start_time = time.time()
    if args.self_forcing:
        logger.info(f"Training DMD2 self-forcing stage for up to {max_steps} steps")
    else:
        logger.info(f"Training DMD2 baseline for up to {max_steps} steps")

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        dataset.set_epoch(epoch)
        dataset.set_aug_ratio(1.0)
        loader_iter = iter(loader)

        completed_steps_in_epoch = max(0, train_steps - epoch * steps_per_epoch)
        completed_steps_in_epoch = min(completed_steps_in_epoch, steps_per_epoch)
        if completed_steps_in_epoch > 0:
            skip_microbatches = completed_steps_in_epoch * args.grad_accum_steps
            for _ in range(skip_microbatches):
                try:
                    next(loader_iter)
                except StopIteration:
                    break
            logger.info(
                f"Skipped {skip_microbatches} microbatches in epoch {epoch} "
                f"after resuming at optimizer step {train_steps}."
            )

        epoch_steps = completed_steps_in_epoch
        while epoch_steps < steps_per_epoch:
            if train_steps >= max_steps:
                break

            microbatches = []
            for _ in range(args.grad_accum_steps):
                try:
                    microbatches.append(next(loader_iter))
                except StopIteration:
                    break
            if len(microbatches) < args.grad_accum_steps:
                break

            do_generator_update = train_steps % args.dfake_gen_update_ratio == 0
            student_head.train()
            fake_score_head.train()
            if discriminator is not None:
                discriminator.train()

            opt_student.zero_grad(set_to_none=True)
            opt_fake.zero_grad(set_to_none=True)
            if discriminator is not None and args.gan_weight > 0:
                opt_disc.zero_grad(set_to_none=True)

            step_logs = {}
            accum_scale = 1.0 / args.grad_accum_steps
            for accum_idx, (real_images, real_classes) in enumerate(microbatches):
                sync_gradients = accum_idx == args.grad_accum_steps - 1
                with ddp_sync_context(
                    [student_head, fake_score_head, discriminator], sync_gradients
                ):
                    real_images = real_images.to(device, non_blocking=True).contiguous(
                        memory_format=torch.channels_last
                    )
                    real_classes = real_classes.to(device, non_blocking=True)
                    real_latents = None
                    if args.prefix_mode == "real" or args.gan_domain in {
                        "latent_grid",
                        "latent_token",
                    }:
                        real_latents = distiller.encode_real_latents(real_images)

                    if args.prefix_mode == "real":
                        fake_classes = real_classes
                    else:
                        fake_classes = torch.randint(
                            0, args.num_classes, (real_images.shape[0],), device=device
                        )

                    with torch.amp.autocast("cuda", dtype=ptdtype):
                        generated_latents, conds = distiller.generate_latents(
                            fake_classes,
                            requires_grad=do_generator_update,
                            prefix_mode=args.prefix_mode,
                            real_latents=real_latents,
                            teacher_sample_steps=args.teacher_sample_steps,
                            teacher_cfg_scale=args.teacher_sample_cfg_scale,
                            teacher_cfg_schedule=args.teacher_sample_cfg_schedule,
                            self_forcing=args.self_forcing,
                            self_forcing_detach_cache=args.self_forcing_detach_cache,
                        )
                        position_indices = sample_position_indices(
                            generated_latents.shape[1],
                            args.token_sample_size,
                            generated_latents.device,
                        )

                        if do_generator_update:
                            dm_loss, dm_logs = distiller.distribution_matching_loss(
                                generated_latents, conds, position_indices
                            )
                            if discriminator is not None and args.gan_weight > 0:
                                fake_inputs, fake_gan_classes, fake_gan_pos = gan_inputs(
                                    args,
                                    distiller,
                                    images=None,
                                    latents=generated_latents,
                                    classes=fake_classes,
                                    position_indices=position_indices,
                                )
                                logits_fake_for_gen = discriminator_for_generator(
                                    discriminator,
                                    fake_inputs,
                                    fake_gan_classes,
                                    fake_gan_pos,
                                )
                                gan_g_loss = generator_loss(
                                    logits_fake_for_gen, args.gan_loss
                                )
                            else:
                                gan_g_loss = torch.tensor(0.0, device=device)
                            student_loss = (
                                args.dm_weight * dm_loss + args.gan_weight * gan_g_loss
                            )
                        else:
                            dm_loss = torch.tensor(0.0, device=device)
                            gan_g_loss = torch.tensor(0.0, device=device)
                            student_loss = torch.tensor(0.0, device=device)

                    if do_generator_update:
                        (student_loss * accum_scale).backward()

                    with torch.amp.autocast("cuda", dtype=ptdtype):
                        fake_score_loss = distiller.fake_score_loss(
                            generated_latents.detach(), conds.detach(), position_indices
                        )
                    (fake_score_loss * accum_scale).backward()

                    if discriminator is not None and args.gan_weight > 0:
                        with torch.no_grad():
                            fake_inputs_for_disc, fake_disc_classes, fake_disc_pos = (
                                gan_inputs(
                                    args,
                                    distiller,
                                    images=None,
                                    latents=generated_latents.detach(),
                                    classes=fake_classes,
                                    position_indices=position_indices,
                                )
                            )
                            if torch.is_tensor(fake_inputs_for_disc):
                                fake_inputs_for_disc = fake_inputs_for_disc.detach()
                            real_inputs_for_disc, real_disc_classes, real_disc_pos = (
                                gan_inputs(
                                    args,
                                    distiller,
                                    images=real_images,
                                    latents=real_latents,
                                    classes=real_classes,
                                    position_indices=position_indices,
                                )
                            )
                        with torch.amp.autocast("cuda", dtype=ptdtype):
                            logits_real = discriminator_forward(
                                discriminator,
                                real_inputs_for_disc,
                                real_disc_classes,
                                real_disc_pos,
                            )
                            logits_fake = discriminator_forward(
                                discriminator,
                                fake_inputs_for_disc,
                                fake_disc_classes,
                                fake_disc_pos,
                            )
                            disc_loss = discriminator_loss(
                                logits_real, logits_fake, args.gan_loss
                            )
                        (disc_loss * accum_scale).backward()
                    else:
                        disc_loss = torch.tensor(0.0, device=device)

                    micro_logs = {
                        "fake_score_loss": fake_score_loss.detach(),
                        "disc_loss": disc_loss.detach(),
                    }
                    if do_generator_update:
                        micro_logs.update(
                            {
                                "student_loss": student_loss.detach(),
                                "dm_loss": dm_loss.detach(),
                                "gan_g_loss": gan_g_loss.detach(),
                                **dm_logs,
                            }
                        )
                    for k, v in micro_logs.items():
                        step_logs[k] = step_logs.get(k, 0.0) + float(v.detach().float())

            if do_generator_update:
                if args.self_forcing:
                    all_reduce_trainable_grads(ar_backbone_parameters(teacher))
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        generator_params, args.max_grad_norm
                    )
                opt_student.step()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    fake_score_head.parameters(), args.max_grad_norm
                )
            opt_fake.step()
            if discriminator is not None and args.gan_weight > 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        discriminator.parameters(), args.max_grad_norm
                    )
                opt_disc.step()

            logs = {k: v / args.grad_accum_steps for k, v in step_logs.items()}
            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v
                running_counts[k] = running_counts.get(k, 0) + 1
            log_count += 1
            train_steps += 1
            epoch_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                elapsed = time.time() - start_time
                parts = []
                for k in sorted(running):
                    stats = torch.tensor(
                        [running[k], running_counts[k]],
                        device=device,
                        dtype=torch.float32,
                    )
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                    avg = (stats[0] / stats[1].clamp_min(1.0)).item()
                    parts.append(f"{k}: {avg:.4f}")
                logger.info(
                    f"(step={train_steps:07d}) "
                    + ", ".join(parts)
                    + f", steps/s: {log_count / max(elapsed, 1e-6):.2f}"
                )
                running = {}
                running_counts = {}
                log_count = 0
                start_time = time.time()

            if (
                rank == 0
                and args.ckpt_every > 0
                and train_steps % args.ckpt_every == 0
            ):
                save_checkpoint(
                    args,
                    epoch,
                    train_steps,
                    teacher,
                    student_head,
                    fake_score_head,
                    discriminator,
                    opt_student,
                    opt_fake,
                    opt_disc,
                    logger,
                )

            if args.preview_every > 0 and train_steps % args.preview_every == 0:
                dist.barrier()
                if rank == 0:
                    save_preview_samples(
                        args,
                        distiller,
                        train_steps,
                        device,
                        ptdtype,
                        logger,
                    )
                dist.barrier()

        save_epoch = epoch + 1 if epoch_steps >= steps_per_epoch else epoch
        if rank == 0:
            save_checkpoint(
                args,
                save_epoch,
                train_steps,
                teacher,
                student_head,
                fake_score_head,
                discriminator,
                opt_student,
                opt_fake,
                opt_disc,
                logger,
                keep_epoch=epoch_steps >= steps_per_epoch,
            )
        dist.barrier()
        if train_steps >= max_steps:
            break

    logger.info("DMD2 distillation done.")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = get_model_args()
    parser.add_argument("--teacher-ckpt", type=str, required=True)
    parser.add_argument("--teacher-no-ema", action="store_true")
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Keep SphereAR torch.compile decorators active during distillation training. Disabled by default.",
    )
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="dmd2_results")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--init-from",
        type=str,
        default="",
        help="Load distillation model weights only and start a fresh run from step 0.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--student-lr", type=float, default=2e-6)
    parser.add_argument("--fake-score-lr", type=float, default=2e-6)
    parser.add_argument("--disc-lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--dm-weight", type=float, default=1.0)
    parser.add_argument("--gan-weight", type=float, default=3e-3)
    parser.add_argument("--dfake-gen-update-ratio", type=int, default=5)
    parser.add_argument(
        "--self-forcing",
        action="store_true",
        help="Use student autoregressive rollout as prefixes and train the AR backbone with the one-step head.",
    )
    detach_group = parser.add_mutually_exclusive_group()
    detach_group.add_argument(
        "--self-forcing-detach-cache",
        action="store_true",
        default=True,
        help="Detach generated prefix tokens and previous KV cache during self-forcing.",
    )
    detach_group.add_argument(
        "--no-self-forcing-detach-cache",
        dest="self_forcing_detach_cache",
        action="store_false",
        help="Allow gradients through generated prefix tokens and previous KV cache during self-forcing.",
    )
    parser.add_argument(
        "--prefix-mode",
        type=str,
        default="teacher_forcing",
        choices=["teacher_forcing", "real"],
    )
    parser.add_argument("--teacher-sample-steps", type=int, default=100)
    parser.add_argument("--teacher-sample-cfg-scale", type=float, default=1.0)
    parser.add_argument(
        "--teacher-sample-cfg-schedule",
        type=str,
        default="linear",
        choices=["linear", "constant"],
    )
    parser.add_argument(
        "--token-sample-size",
        type=int,
        default=-1,
        help="Number of raster positions used by DM/fake-score losses per step; <=0 uses all positions.",
    )
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument(
        "--cfg-schedule", type=str, default="linear", choices=["linear", "constant"]
    )
    parser.add_argument(
        "--gan-domain",
        type=str,
        default="image",
        choices=["image", "latent_grid", "latent_token", "none"],
    )
    parser.add_argument(
        "--disc-type",
        type=str,
        default="resnet",
        choices=["resnet", "patchgan", "stylegan"],
    )
    parser.add_argument("--disc-dim", type=int, default=64)
    parser.add_argument(
        "--gan-loss", type=str, default="hinge", choices=["hinge", "non-saturating"]
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--keep-freq", type=int, default=10)
    parser.add_argument("--preview-every", type=int, default=-1)
    parser.add_argument("--preview-num", type=int, default=16)
    parser.add_argument("--preview-batch-size", type=int, default=0)
    parser.add_argument("--preview-dir", type=str, default="")
    parser.add_argument("--preview-seed", type=int, default=1234)
    parser.add_argument(
        "--mixed-precision", type=str, default="bf16", choices=["none", "bf16"]
    )
    args = parser.parse_args()
    world_size = int(os.environ["WORLD_SIZE"])
    assert args.grad_accum_steps >= 1
    assert args.dfake_gen_update_ratio >= 1
    assert args.global_batch_size % (world_size * args.grad_accum_steps) == 0
    main(args)
