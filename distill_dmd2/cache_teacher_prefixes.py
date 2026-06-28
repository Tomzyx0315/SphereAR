import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist


def _identity_compile(fn=None, *args, **kwargs):
    if fn is None:
        return lambda inner_fn: inner_fn
    return fn


if "--compile-model" not in sys.argv:
    torch.compile = _identity_compile

from SphereAR.model import create_model, get_model_args
from SphereAR.dataset import ImageNetTarDataset
from SphereAR.utils import requires_grad
from distill_dmd2.distiller import SphereARDMD2Distiller
from torchvision.datasets import ImageFolder


SHARD_SIZE = 4096
CACHE_DTYPE = np.float16
CACHE_DTYPE_NAME = "float16"


def init_distributed_mode():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    return rank, dist.get_world_size(), device


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


def data_labels(data_path):
    if data_path.endswith(".tar"):
        dataset = ImageNetTarDataset(data_path)
        labels = [label for _offset, _size, label in dataset.files]
    else:
        dataset = ImageFolder(data_path)
        labels = [label for _path, label in dataset.samples]
    return torch.tensor(labels, dtype=torch.long)


def build_cache_labels(args):
    if not args.data_path:
        raise ValueError("--data-path is required for full-dataset teacher caching.")
    return data_labels(args.data_path)


def build_cache_labels_distributed(args, rank, device):
    if rank == 0:
        labels = build_cache_labels(args)
        length = torch.tensor([labels.numel()], device=device, dtype=torch.long)
    else:
        labels = None
        length = torch.zeros(1, device=device, dtype=torch.long)

    dist.broadcast(length, src=0)
    if rank != 0:
        labels = torch.empty(int(length.item()), dtype=torch.long)
    labels_device = labels.to(device)
    dist.broadcast(labels_device, src=0)
    return labels_device.cpu()


def rank_slice(total, rank, world_size):
    per_rank = (total + world_size - 1) // world_size
    start = min(rank * per_rank, total)
    end = min(start + per_rank, total)
    return start, end


def save_shard(cache_dir, rank, shard_idx, latents, labels):
    stem = f"rank{rank:04d}_shard{shard_idx:06d}"
    latent_name = f"{stem}_latents.npy"
    label_name = f"{stem}_labels.npy"
    latent_path = os.path.join(cache_dir, latent_name)
    label_path = os.path.join(cache_dir, label_name)
    np.save(latent_path, latents)
    np.save(label_path, labels)
    return {
        "latents": latent_name,
        "labels": label_name,
        "num_samples": int(labels.shape[0]),
    }


def collect_shards(cache_dir):
    shards = []
    for name in sorted(os.listdir(cache_dir)):
        if not name.endswith("_labels.npy"):
            continue
        stem = name[: -len("_labels.npy")]
        latent_name = f"{stem}_latents.npy"
        latent_path = os.path.join(cache_dir, latent_name)
        label_path = os.path.join(cache_dir, name)
        if not os.path.exists(latent_path):
            raise FileNotFoundError(f"Missing latent shard for {label_path}")
        labels = np.load(label_path, mmap_mode="r")
        shards.append(
            {
                "latents": latent_name,
                "labels": name,
                "num_samples": int(labels.shape[0]),
            }
        )
    return shards


def check_cache_dir_available(cache_dir):
    if not os.path.exists(cache_dir):
        return
    generated = [
        name
        for name in os.listdir(cache_dir)
        if name == "meta.json"
        or name.endswith("_latents.npy")
        or name.endswith("_labels.npy")
    ]
    if generated:
        raise ValueError(
            f"{cache_dir} already contains teacher cache files. Use a new "
            "directory to avoid mixing shards from different runs."
        )


def main(args):
    assert torch.cuda.is_available(), "Teacher prefix caching requires CUDA."
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rank, world_size, device = init_distributed_mode()
    check_cache_dir_available(args.cache_dir)
    os.makedirs(args.cache_dir, exist_ok=True)
    dist.barrier()
    seed = args.seed * world_size + rank
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)

    all_labels = build_cache_labels_distributed(args, rank, device)
    total_samples = len(all_labels)
    start, end = rank_slice(total_samples, rank, world_size)
    rank_labels = all_labels[start:end]

    if rank == 0:
        print(
            f"Caching {total_samples} teacher prefixes to {args.cache_dir} "
            f"across {world_size} ranks."
        )
    print(f"Rank {rank} generating labels [{start}, {end})")

    teacher = load_teacher(args, device)
    distiller = SphereARDMD2Distiller(
        teacher=teacher,
        student_head=None,
        fake_score_head=None,
        discriminator=None,
        cfg_scale=1.0,
    )

    ptdtype = {"none": torch.float32, "bf16": torch.bfloat16}[args.mixed_precision]
    pending_latents = []
    pending_labels = []
    shard_idx = 0
    produced = 0
    start_time = time.time()

    for offset in range(0, len(rank_labels), args.batch_size):
        labels = rank_labels[offset : offset + args.batch_size].to(device)
        with torch.amp.autocast("cuda", dtype=ptdtype):
            latents = distiller.sample_teacher_latents(
                labels,
                sample_steps=args.teacher_sample_steps,
                cfg_scale=args.teacher_sample_cfg_scale,
                cfg_schedule=args.teacher_sample_cfg_schedule,
            )
        pending_latents.append(latents.cpu().float().numpy().astype(CACHE_DTYPE))
        pending_labels.append(labels.cpu().numpy().astype(np.int64))
        produced += int(labels.shape[0])

        if sum(x.shape[0] for x in pending_labels) >= SHARD_SIZE:
            shard_latents = np.concatenate(pending_latents, axis=0)
            shard_labels = np.concatenate(pending_labels, axis=0)
            save_shard(
                cache_dir=args.cache_dir,
                rank=rank,
                shard_idx=shard_idx,
                latents=shard_latents,
                labels=shard_labels,
            )
            print(
                f"Rank {rank} saved shard {shard_idx} with {shard_labels.shape[0]} "
                f"samples; produced {produced}/{len(rank_labels)} in "
                f"{time.time() - start_time:.1f}s"
            )
            pending_latents = []
            pending_labels = []
            shard_idx += 1

    if pending_labels:
        shard_latents = np.concatenate(pending_latents, axis=0)
        shard_labels = np.concatenate(pending_labels, axis=0)
        save_shard(
            cache_dir=args.cache_dir,
            rank=rank,
            shard_idx=shard_idx,
            latents=shard_latents,
            labels=shard_labels,
        )
        print(
            f"Rank {rank} saved final shard {shard_idx} with "
            f"{shard_labels.shape[0]} samples."
        )

    dist.barrier()
    if rank == 0:
        shards = collect_shards(args.cache_dir)
        meta = {
            "version": 1,
            "format": "teacher_prefix_latents_npy",
            "model": args.model,
            "teacher_ckpt": os.path.abspath(args.teacher_ckpt),
            "teacher_no_ema": args.teacher_no_ema,
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "latent_dim": args.latent_dim,
            "num_classes": args.num_classes,
            "cls_token_num": args.cls_token_num,
            "teacher_sample_steps": args.teacher_sample_steps,
            "teacher_sample_cfg_scale": args.teacher_sample_cfg_scale,
            "teacher_sample_cfg_schedule": args.teacher_sample_cfg_schedule,
            "cache_dtype": CACHE_DTYPE_NAME,
            "num_samples": int(sum(shard["num_samples"] for shard in shards)),
            "data_path": os.path.abspath(args.data_path) if args.data_path else "",
            "seed": args.seed,
            "shards": shards,
        }
        meta_path = os.path.join(args.cache_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        bytes_per_value = np.dtype(CACHE_DTYPE).itemsize
        latent_bytes = meta["num_samples"] * ((args.image_size // args.patch_size) ** 2)
        latent_bytes *= args.latent_dim * bytes_per_value
        print(f"Wrote {meta_path}")
        print(f"Approx latent storage: {latent_bytes / (1024 ** 3):.2f} GiB")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = get_model_args()
    parser.add_argument("--teacher-ckpt", type=str, required=True)
    parser.add_argument("--teacher-no-ema", action="store_true")
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Keep SphereAR torch.compile decorators active during cache generation.",
    )
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--teacher-sample-steps", type=int, default=100)
    parser.add_argument("--teacher-sample-cfg-scale", type=float, default=4.6)
    parser.add_argument(
        "--teacher-sample-cfg-schedule",
        type=str,
        default="linear",
        choices=["linear", "constant"],
    )
    parser.add_argument(
        "--mixed-precision", type=str, default="bf16", choices=["none", "bf16"]
    )
    main(parser.parse_args())
