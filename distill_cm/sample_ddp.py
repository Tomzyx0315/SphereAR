import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch


def _identity_compile(fn=None, *args, **kwargs):
    if fn is None:
        return lambda inner_fn: inner_fn
    return fn


if "--compile-model" not in sys.argv:
    torch.compile = _identity_compile

import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from SphereAR.model import create_model, get_model_args
from SphereAR.utils import requires_grad
from distill_cm.distiller import SphereARConsistencyDistiller, karras_sigmas
from distill_cm.heads import ConsistencyHead


def safe_path_part(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def checkpoint_string(path):
    ckpt_path = Path(path)
    parent = safe_path_part(ckpt_path.parent.name)
    stem = safe_path_part(ckpt_path.stem)
    return f"{parent}-{stem}" if parent else stem


def create_npz_from_sample_folder(sample_dir, num=50_000, keep_pngs=False):
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    if not keep_pngs:
        shutil.rmtree(sample_dir)
    return npz_path


def load_teacher(args, device):
    teacher = create_model(args, device)
    checkpoint = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    if "ema" in checkpoint and not args.teacher_no_ema:
        state = checkpoint["ema"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        raise ValueError(f"Cannot find model weights in {args.teacher_ckpt}")
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    requires_grad(teacher, False)
    return teacher


def parse_sampling_sigmas(args, device):
    if args.sampling_sigmas:
        values = [float(part) for part in args.sampling_sigmas.split(",") if part]
        if not values:
            raise ValueError("--sampling-sigmas did not contain any values.")
        return values
    if args.num_consistency_steps <= 1:
        return [args.sigma_max]
    sigmas = karras_sigmas(
        args.num_consistency_steps,
        args.sigma_min,
        args.sigma_max,
        args.rho,
        device,
    )
    return [float(value.item()) for value in sigmas]


def main(args):
    assert torch.cuda.is_available(), "DDP sampling requires CUDA."
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    seed = args.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)

    teacher = load_teacher(args, device)
    student_head = ConsistencyHead(teacher.head).to(device)
    checkpoint = torch.load(args.distill_ckpt, map_location="cpu", weights_only=False)
    if args.head == "target" and "target_head" in checkpoint:
        student_key = "target_head"
    else:
        student_key = "student_head"
    student_head.load_state_dict(checkpoint[student_key], strict=True)
    student_head.eval()

    distiller = SphereARConsistencyDistiller(
        teacher=teacher,
        student_head=student_head,
        target_head=student_head,
        cfg_scale=args.cfg_scale,
        cfg_schedule=args.cfg_schedule,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho=args.rho,
        sigma_data=args.sigma_data,
        weight_schedule=args.weight_schedule,
        loss_norm="l2",
        normalize_denoised=args.normalize_denoised,
    )
    sampling_sigmas = parse_sampling_sigmas(args, device)

    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = checkpoint_string(args.distill_ckpt)
    sigma_name = "-".join(f"{sigma:g}" for sigma in sampling_sigmas)
    folder_name = (
        f"{model_string_name}-cm-{ckpt_string_name}-size-{args.image_size}-"
        f"cfg-{args.cfg_scale}-{args.cfg_schedule}-sigmas-{safe_path_part(sigma_name)}-"
        f"seed-{args.seed}-n-{args.num_fid_samples}"
    )
    if args.teacher_no_ema:
        folder_name += "-teacher-noema"
    if args.sample_name:
        folder_name = safe_path_part(args.sample_name)
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"

    if os.path.isfile(sample_folder_dir + ".npz"):
        if rank == 0:
            print(f"Found {sample_folder_dir}.npz, skipping sampling.")
        dist.barrier()
        dist.destroy_process_group()
        return
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
        print(f"Sampling sigmas: {sampling_sigmas}")
    dist.barrier()

    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(
        math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size
    )
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    samples_needed_this_gpu = total_samples // dist.get_world_size()
    iterations = samples_needed_this_gpu // n
    total = 0
    start_time = time.time()
    precision = None if args.mixed_precision == "none" else torch.bfloat16

    for _ in tqdm(range(iterations), desc="Sampling"):
        class_id = torch.randint(0, args.num_classes, (n,), device=device)
        context = (
            torch.amp.autocast("cuda", dtype=precision)
            if precision is not None
            else torch.no_grad()
        )
        with context:
            latents = distiller.generate_latents_autoregressive(
                class_id, sampling_sigmas=sampling_sigmas
            )
            samples = distiller.decode_latents(latents)
        samples = (
            torch.clamp(127.5 * samples + 128.0, 0, 255)
            .permute(0, 2, 3, 1)
            .to("cpu", dtype=torch.uint8)
            .numpy()
        )
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size
        print(
            f"Rank {rank} sampled {total} images, cost {time.time() - start_time:.2f}s"
        )

    dist.barrier()
    if rank == 0 and args.to_npz:
        create_npz_from_sample_folder(
            sample_folder_dir, args.num_fid_samples, keep_pngs=args.keep_pngs
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = get_model_args()
    parser.add_argument("--teacher-ckpt", type=str, required=True)
    parser.add_argument("--teacher-no-ema", action="store_true")
    parser.add_argument("--distill-ckpt", type=str, required=True)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--sample-name", type=str, default="")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument(
        "--cfg-schedule", type=str, default="linear", choices=["linear", "constant"]
    )
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sigma-data", type=float, default=1.0)
    parser.add_argument(
        "--sampling-sigmas",
        type=str,
        default="",
        help="Comma-separated descending sigmas. Empty uses --num-consistency-steps.",
    )
    parser.add_argument("--num-consistency-steps", type=int, default=1)
    parser.add_argument(
        "--weight-schedule",
        type=str,
        default="uniform",
        choices=["uniform", "snr", "snr+1", "karras", "truncated-snr"],
    )
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
    parser.add_argument(
        "--head",
        type=str,
        default="target",
        choices=["target", "student"],
        help="Checkpoint head used for sampling; target is the EMA consistency head.",
    )
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument(
        "--mixed-precision", type=str, default="bf16", choices=["none", "bf16"]
    )
    parser.add_argument("--to-npz", action="store_true")
    parser.add_argument("--keep-pngs", action="store_true")
    main(parser.parse_args())
