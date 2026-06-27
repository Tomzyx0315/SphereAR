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
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from SphereAR.model import create_model, get_model_args
from SphereAR.utils import requires_grad
from distill_dmd2.ar_utils import load_ar_backbone_state_dict
from distill_dmd2.distiller import SphereARDMD2Distiller
from distill_dmd2.heads import OneStepHead


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
    student_head = OneStepHead(teacher.head).to(device)
    checkpoint = torch.load(args.distill_ckpt, map_location="cpu", weights_only=False)
    if "ar_backbone" in checkpoint:
        load_ar_backbone_state_dict(teacher, checkpoint["ar_backbone"])
    student_head.load_state_dict(checkpoint["student_head"], strict=True)
    student_head.eval()

    distiller = SphereARDMD2Distiller(
        teacher=teacher,
        student_head=student_head,
        fake_score_head=student_head,
        discriminator=None,
        cfg_scale=args.cfg_scale,
        cfg_schedule=args.cfg_schedule,
    )

    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = checkpoint_string(args.distill_ckpt)
    folder_name = (
        f"{model_string_name}-dmd2-{ckpt_string_name}-size-{args.image_size}-"
        f"cfg-{args.cfg_scale}-{args.cfg_schedule}-seed-{args.seed}-"
        f"n-{args.num_fid_samples}"
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
    precision = {"none": torch.float32, "bf16": torch.bfloat16}[args.mixed_precision]

    for _ in tqdm(range(iterations), desc="Sampling"):
        class_id = torch.randint(0, args.num_classes, (n,), device=device)
        with torch.amp.autocast("cuda", dtype=precision):
            latents, _ = distiller.generate_latents_autoregressive(class_id)
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
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument(
        "--mixed-precision", type=str, default="bf16", choices=["none", "bf16"]
    )
    parser.add_argument("--to-npz", action="store_true")
    parser.add_argument("--keep-pngs", action="store_true")
    main(parser.parse_args())
