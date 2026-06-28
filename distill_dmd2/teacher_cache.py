import bisect
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class TeacherPrefixCacheDataset(Dataset):
    """Memory-mapped teacher prefix latent cache."""

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        meta_path = os.path.join(cache_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Teacher prefix cache meta not found: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.shards = self.meta.get("shards", [])
        if not self.shards:
            raise ValueError(f"No shards found in teacher prefix cache: {cache_dir}")

        self.cumulative = []
        total = 0
        for shard in self.shards:
            total += int(shard["num_samples"])
            self.cumulative.append(total)
        self.num_samples = total
        self._latents = None
        self._labels = None

    def _open_shards(self):
        if self._latents is not None:
            return
        self._latents = []
        self._labels = []
        for shard in self.shards:
            latent_path = os.path.join(self.cache_dir, shard["latents"])
            label_path = os.path.join(self.cache_dir, shard["labels"])
            self._latents.append(np.load(latent_path, mmap_mode="r"))
            self._labels.append(np.load(label_path, mmap_mode="r"))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        self._open_shards()
        shard_idx = bisect.bisect_right(self.cumulative, idx)
        shard_start = 0 if shard_idx == 0 else self.cumulative[shard_idx - 1]
        local_idx = idx - shard_start
        latent = np.array(self._latents[shard_idx][local_idx], copy=True)
        label = int(self._labels[shard_idx][local_idx])
        return torch.from_numpy(latent), torch.tensor(label, dtype=torch.long)


def load_cache_meta(cache_dir):
    meta_path = os.path.join(cache_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_teacher_cache(cache_dir, args):
    meta = load_cache_meta(cache_dir)
    expected = {
        "model": args.model,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "latent_dim": args.latent_dim,
        "num_classes": args.num_classes,
        "cls_token_num": args.cls_token_num,
    }
    mismatches = []
    for key, value in expected.items():
        if meta.get(key) != value:
            mismatches.append(f"{key}: cache={meta.get(key)} train={value}")
    if mismatches:
        raise ValueError(
            "Teacher prefix cache does not match training configuration: "
            + "; ".join(mismatches)
        )
    return meta
