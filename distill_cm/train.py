import math
import os
import sys
import time
from contextlib import ExitStack, nullcontext
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
from distill_cm.distiller import SphereARConsistencyDistiller, unwrap_model
from distill_cm.heads import ConsistencyHead


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


def autocast_context(dtype):
    if dtype is None:
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=dtype)


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


@torch.no_grad()
def update_target_ema(target_head, student_head, rate):
    target = unwrap_model(target_head)
    student = unwrap_model(student_head)
    for target_param, student_param in zip(target.parameters(), student.parameters()):
        target_param.detach().mul_(rate).add_(student_param.detach(), alpha=1.0 - rate)


def copy_student_to_target(target_head, student_head):
    unwrap_model(target_head).load_state_dict(
        unwrap_model(student_head).state_dict(), strict=True
    )


def save_checkpoint(
    args,
    epoch,
    step,
    student_head,
    target_head,
    optimizer,
    logger,
    keep_epoch=False,
):
    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint = {
        "student_head": unwrap_model(student_head).state_dict(),
        "target_head": unwrap_model(target_head).state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "args": args,
    }
    path = os.path.join(args.results_dir, "last.pt")
    torch.save(checkpoint, path)
    if keep_epoch and args.keep_freq > 0 and epoch > 0 and epoch % args.keep_freq == 0:
        torch.save(checkpoint, os.path.join(args.results_dir, f"epoch_{epoch}.pt"))
    logger.info(f"Saved consistency distillation checkpoint to {path}")


def load_init_from(args, student_head, target_head, logger):
    if not args.init_from:
        return
    checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=False)
    unwrap_model(student_head).load_state_dict(checkpoint["student_head"], strict=True)
    if "target_head" in checkpoint:
        unwrap_model(target_head).load_state_dict(checkpoint["target_head"], strict=True)
    else:
        copy_student_to_target(target_head, student_head)
    logger.info(f"Initialized consistency weights from {args.init_from}.")


def load_resume(args, student_head, target_head, optimizer, logger):
    if not args.resume:
        return 0, 0
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    for key in ("student_head", "optimizer", "epoch", "step"):
        if key not in checkpoint:
            raise KeyError(f"--resume checkpoint {args.resume} is missing {key}.")
    unwrap_model(student_head).load_state_dict(checkpoint["student_head"], strict=True)
    if "target_head" in checkpoint:
        unwrap_model(target_head).load_state_dict(checkpoint["target_head"], strict=True)
    else:
        copy_student_to_target(target_head, student_head)
    optimizer.load_state_dict(checkpoint["optimizer"])
    logger.info(f"Resumed consistency distillation from {args.resume}.")
    return checkpoint["epoch"], checkpoint["step"]


def ddp_sync_context(modules, sync_gradients):
    stack = ExitStack()
    if not sync_gradients:
        for module in modules:
            if module is not None and hasattr(module, "no_sync"):
                stack.enter_context(module.no_sync())
    return stack


def sample_position_indices(seq_len, sample_size, device):
    if sample_size <= 0 or sample_size >= seq_len:
        return None
    return torch.randperm(seq_len, device=device)[:sample_size].sort().values


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
                with autocast_context(ptdtype):
                    latents = distiller.generate_latents_autoregressive(class_id)
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
        distiller.teacher.train(teacher_was_training)
        student.train(student_was_training)


def main(args):
    assert torch.cuda.is_available(), "Consistency distillation requires CUDA."
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
    if args.num_scales < 2:
        raise ValueError("--num-scales must be >= 2.")

    teacher = load_teacher(args, device, logger)
    student_head = ConsistencyHead(teacher.head).to(device)
    target_head = ConsistencyHead(teacher.head).to(device)
    copy_student_to_target(target_head, student_head)
    for param in target_head.parameters():
        param.requires_grad_(False)
    target_head.eval()

    student_head = DDP(student_head, device_ids=[args.gpu])
    optimizer = create_optimizer(
        student_head.parameters(), args.lr, args.weight_decay
    )

    load_init_from(args, student_head, target_head, logger)
    start_epoch, train_steps = load_resume(
        args, student_head, target_head, optimizer, logger
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

    distiller = SphereARConsistencyDistiller(
        teacher=teacher,
        student_head=student_head,
        target_head=target_head,
        cfg_scale=args.cfg_scale,
        cfg_schedule=args.cfg_schedule,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho=args.rho,
        sigma_data=args.sigma_data,
        weight_schedule=args.weight_schedule,
        loss_norm=args.loss_norm,
        normalize_denoised=args.normalize_denoised,
        loss_eps=args.loss_eps,
    )

    ptdtype = None if args.mixed_precision == "none" else torch.bfloat16
    steps_per_epoch = int(len(dataset) / args.global_batch_size)
    max_steps = args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch

    running = {}
    running_counts = {}
    log_count = 0
    start_time = time.time()
    logger.info(f"Training consistency distillation for up to {max_steps} steps")

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
                    real_batch = next(loader_iter)
                except StopIteration:
                    break
                microbatches.append(real_batch)
            if len(microbatches) < args.grad_accum_steps:
                break

            student_head.train()
            optimizer.zero_grad(set_to_none=True)
            step_logs = {}
            accum_scale = 1.0 / args.grad_accum_steps

            for accum_idx, real_batch in enumerate(microbatches):
                real_images, real_classes = real_batch
                sync_gradients = accum_idx == args.grad_accum_steps - 1
                with ddp_sync_context([student_head], sync_gradients):
                    real_images = real_images.to(device, non_blocking=True).contiguous(
                        memory_format=torch.channels_last
                    )
                    real_classes = real_classes.to(device, non_blocking=True)
                    real_latents = distiller.encode_real_latents(real_images)

                    with autocast_context(ptdtype):
                        clean_latents, conds = distiller.build_real_training_batch(
                            real_classes,
                            real_latents,
                        )
                        position_indices = sample_position_indices(
                            clean_latents.shape[1],
                            args.token_sample_size,
                            clean_latents.device,
                        )
                        loss, loss_logs = distiller.consistency_loss(
                            clean_latents,
                            conds,
                            num_scales=args.num_scales,
                            position_indices=position_indices,
                        )

                    (loss * accum_scale).backward()
                    micro_logs = {"cm_loss": loss.detach(), **loss_logs}
                    for key, value in micro_logs.items():
                        step_logs[key] = step_logs.get(key, 0.0) + float(
                            value.detach().float()
                        )

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    student_head.parameters(), args.max_grad_norm
                )
            optimizer.step()
            update_target_ema(target_head, student_head, args.target_ema)

            logs = {key: value / args.grad_accum_steps for key, value in step_logs.items()}
            for key, value in logs.items():
                running[key] = running.get(key, 0.0) + value
                running_counts[key] = running_counts.get(key, 0) + 1
            log_count += 1
            train_steps += 1
            epoch_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                elapsed = time.time() - start_time
                parts = []
                for key in sorted(running):
                    stats = torch.tensor(
                        [running[key], running_counts[key]],
                        device=device,
                        dtype=torch.float32,
                    )
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                    avg = (stats[0] / stats[1].clamp_min(1.0)).item()
                    parts.append(f"{key}: {avg:.4f}")
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
                    student_head,
                    target_head,
                    optimizer,
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
                student_head,
                target_head,
                optimizer,
                logger,
                keep_epoch=epoch_steps >= steps_per_epoch,
            )
        dist.barrier()
        if train_steps >= max_steps:
            break

    logger.info("Consistency distillation done.")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = get_model_args()
    parser.add_argument("--teacher-ckpt", type=str, required=True)
    parser.add_argument("--teacher-no-ema", action="store_true")
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Keep SphereAR torch.compile decorators active during training.",
    )
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="cm_results")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--init-from", type=str, default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--token-sample-size", type=int, default=-1)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument(
        "--cfg-schedule", type=str, default="linear", choices=["linear", "constant"]
    )
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sigma-data", type=float, default=1.0)
    parser.add_argument("--num-scales", type=int, default=40)
    parser.add_argument("--target-ema", type=float, default=0.95)
    parser.add_argument(
        "--weight-schedule",
        type=str,
        default="uniform",
        choices=["uniform", "snr", "snr+1", "karras", "truncated-snr"],
    )
    parser.add_argument(
        "--loss-norm",
        type=str,
        default="l2",
        choices=["l1", "l2", "pseudo-huber"],
    )
    parser.add_argument("--loss-eps", type=float, default=1e-3)
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument(
        "--normalize-denoised",
        dest="normalize_denoised",
        action="store_true",
        default=True,
    )
    norm_group.add_argument(
        "--no-normalize-denoised",
        dest="normalize_denoised",
        action="store_false",
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
    assert args.global_batch_size % (world_size * args.grad_accum_steps) == 0
    main(args)
