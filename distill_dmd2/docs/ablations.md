# DMD2 Distillation Ablations

This file lists the ablations currently exposed by `distill_dmd2/train.py` and
`distill_dmd2/sample_ddp.py`.

## Gradient Accumulation

Training supports gradient accumulation with:

```bash
--grad-accum-steps 1
--grad-accum-steps 2
--grad-accum-steps 4
```

`--global-batch-size` is the effective batch processed by one optimizer step
across all ranks:

```text
per_gpu_microbatch = global_batch_size / (world_size * grad_accum_steps)
```

Each optimizer step runs `grad_accum_steps` full microbatch passes through
teacher-prefix generation, student generation, fake-score update, and
discriminator update. Losses are divided by `grad_accum_steps` before backward.
The dataloader drops incomplete effective batches at the end of an epoch.

## Current Default Baseline

The default distillation baseline is:

```bash
--prefix-mode teacher_forcing
--teacher-sample-steps 100
--teacher-sample-cfg-scale 4.6
--teacher-sample-cfg-schedule linear
--token-sample-size 256
--cfg-scale 1.0
--cfg-schedule linear
--gan-domain image
--disc-type resnet
--gan-loss hinge
--dfake-gen-update-ratio 5
--dm-weight 1.0
--gan-weight 3e-3
```

The README command uses all 256 raster positions for 256x256, `patch_size=16`.
`-1` is equivalent to all raster positions.

## Training Ablations

### Teacher Checkpoint

Flags:

```bash
--teacher-ckpt /path/to/SphereAR_B.pt
--teacher-no-ema
```

Behavior:

- By default, the loader uses `checkpoint["ema"]` when present.
- `--teacher-no-ema` uses `checkpoint["model"]` when both exist.

Suggested comparisons:

```bash
# EMA teacher
--teacher-ckpt $TEACHER_CKPT

# Raw model teacher
--teacher-ckpt $TEACHER_CKPT --teacher-no-ema
```

### Model Scale

Inherited model flags:

```bash
--model SphereAR-B
--model SphereAR-L
--model SphereAR-H
--image-size 256
--image-size 512
--patch-size 16
--latent-dim 16
--num-classes 1000
--cls-token-num 16
```

Use the model flag that matches the teacher checkpoint.

### Prefix Construction

Flags:

```bash
--prefix-mode teacher_forcing
--prefix-mode real
--self-forcing
--self-forcing-detach-cache
--no-self-forcing-detach-cache
```

Modes:

- `teacher_forcing`: teacher samples a clean latent sequence first; the frozen
  AR trunk builds conditions from teacher clean prefixes.
- `real`: VAE encodes the current real image batch; the frozen AR trunk builds
  conditions from real-image latent prefixes.
- `--self-forcing`: second-stage mode used with `--prefix-mode teacher_forcing`.
  The student rolls out tokens autoregressively, position `i` is conditioned on
  generated tokens `< i`, and the AR backbone is trained together with the
  one-step head. The VAE and teacher diffusion head remain frozen.
- `--self-forcing` detaches previous generated tokens and previous KV cache
  entries by default. The generated prefix values are still student samples,
  but gradients are truncated across AR history.
- `--no-self-forcing-detach-cache`: used with `--self-forcing` to allow
  gradients through previous generated tokens and previous KV cache entries.

Suggested comparisons:

```bash
# Main baseline
--prefix-mode teacher_forcing

# Clean real-prefix diagnostic
--prefix-mode real

# On-policy prefix second stage
--prefix-mode teacher_forcing --self-forcing

# Full-history-gradient on-policy prefix second stage
--prefix-mode teacher_forcing --self-forcing --no-self-forcing-detach-cache
```

### Teacher Prefix Sampling

Flags:

```bash
--teacher-sample-steps 100
--teacher-sample-cfg-scale 1.0
--teacher-sample-cfg-schedule linear
--teacher-sample-cfg-schedule constant
```

These affect the teacher clean tokens used as prefixes in `teacher_forcing`
mode. They are ignored by `--self-forcing`, because self-forcing prefixes come
from the student autoregressive rollout.

Suggested comparisons:

```bash
# Unguided teacher prefixes
--teacher-sample-cfg-scale 1.0

# Guided teacher prefixes
--teacher-sample-cfg-scale 4.6

# Cheaper teacher sampling
--teacher-sample-steps 25
--teacher-sample-steps 50
--teacher-sample-steps 100
```

### Distillation CFG

Flags:

```bash
--cfg-scale 1.0
--cfg-scale 4.5
--cfg-schedule linear
--cfg-schedule constant
```

This CFG is used inside the distillation loss path:

- conditional/null score combination for the teacher and fake score heads
- conditional/null student token prediction before VAE latent normalization

Suggested comparisons:

```bash
# Main baseline
--cfg-scale 1.0

# Guided distillation ablation
--cfg-scale 4.5 --cfg-schedule linear
--cfg-scale 4.5 --cfg-schedule constant
```

### Raster Position Sampling

Flags:

```bash
--token-sample-size -1
--token-sample-size 64
--token-sample-size 128
```

Behavior:

- `<= 0` or `>= seq_len`: use all raster positions.
- Positive values below `seq_len`: randomly sample that many positions per step.
- The sampled positions affect distribution matching and fake-score losses.
- For `--gan-domain latent_token`, the sampled positions also define the real
  and fake tokens seen by the token discriminator.
- For `--gan-domain image` and `--gan-domain latent_grid`, the GAN branch uses
  the full generated latent grid.

Suggested comparisons for 256x256, `patch_size=16`, `seq_len=256`:

```bash
--token-sample-size 32
--token-sample-size 64
--token-sample-size 128
--token-sample-size -1
```

### Loss Weights

Flags:

```bash
--dm-weight 1.0
--gan-weight 3e-3
```

Suggested comparisons:

```bash
--dm-weight 1.0 --gan-weight 1e-3
--dm-weight 1.0 --gan-weight 3e-3
--dm-weight 1.0 --gan-weight 1e-2
```

### DMD2 Update Ratio

Flag:

```bash
--dfake-gen-update-ratio 5
```

Behavior:

- fake score head updates every step
- discriminator updates every step
- student generator updates every `dfake_gen_update_ratio` steps

Suggested comparisons:

```bash
--dfake-gen-update-ratio 1
--dfake-gen-update-ratio 5
--dfake-gen-update-ratio 10
```

### GAN Domain and Discriminator

Flags:

```bash
--gan-domain image
--gan-domain latent_grid
--gan-domain latent_token
--gan-domain none
--disc-type resnet
--disc-type patchgan
--disc-type stylegan
--disc-dim 64
--gan-loss hinge
--gan-loss non-saturating
```

GAN domains:

- `image`: fake samples are decoded by the frozen VAE and discriminated against
  real images.
- `latent_grid`: fake full latent grids are discriminated against VAE-encoded
  real latent grids.
- `latent_token`: sampled fake latent tokens are discriminated against sampled
  VAE-encoded real tokens with class and raster-position conditioning.
- `none`: disables GAN loss and discriminator updates.

Discriminator choices for `--gan-domain image`:

- `resnet`: class-conditional spectral-norm projection ResNet discriminator.
- `patchgan`: PatchGAN discriminator from the original SphereAR GAN utilities.
- `stylegan`: StyleGAN-like discriminator from the original SphereAR GAN
  utilities.

`latent_grid` and `latent_token` use their own projection discriminators and
ignore `--disc-type`.

Suggested comparisons:

```bash
--gan-domain image --disc-type resnet --gan-loss hinge
--gan-domain latent_grid --gan-loss hinge
--gan-domain latent_token --token-sample-size 64 --gan-loss hinge
--gan-domain none --gan-weight 0
--disc-type resnet --gan-loss hinge
--disc-type resnet --gan-loss non-saturating
--disc-type resnet --disc-dim 32
--disc-type resnet --disc-dim 64
--disc-type resnet --disc-dim 128
```

### Optimizers

Flags:

```bash
--student-lr 2e-6
--fake-score-lr 2e-6
--disc-lr 2e-6
--weight-decay 0.01
--max-grad-norm 10.0
```

Suggested comparisons:

```bash
--student-lr 1e-6 --fake-score-lr 1e-6 --disc-lr 1e-6
--student-lr 2e-6 --fake-score-lr 2e-6 --disc-lr 2e-6
--student-lr 5e-6 --fake-score-lr 5e-6 --disc-lr 2e-6

# Self-forcing fine-tuning from a stage-1 checkpoint
--student-lr 5e-7 --fake-score-lr 1e-6 --disc-lr 1e-6

--max-grad-norm 1.0
--max-grad-norm 10.0
```

### Precision

Flag:

```bash
--mixed-precision bf16
--mixed-precision none
```

Suggested comparisons:

```bash
--mixed-precision bf16
--mixed-precision none
```

### Torch Compile

Flag:

```bash
--compile-model
```

Behavior:

- Distillation training disables the SphereAR `torch.compile` decorators by
  default for both stage 1 and stage 2.
- `--compile-model` keeps those decorators active as an experimental ablation.
- This ablation currently has known bugs in the distillation training path, so
  the recommended commands leave it off.

### Batch, Schedule, Logging, Checkpointing

Flags:

```bash
--global-batch-size 64
--grad-accum-steps 1
--epochs 100
--max-steps -1
--global-seed 0
--num-workers 8
--log-every 50
--ckpt-every 1000
--keep-freq 10
--resume /path/to/last.pt
--preview-every -1
--preview-num 16
--preview-batch-size 0
--preview-dir ""
--preview-seed 1234
--init-from ""
```

Suggested comparisons:

```bash
--global-batch-size 64
--global-batch-size 128
--global-batch-size 256 --grad-accum-steps 2
--global-batch-size 512 --grad-accum-steps 4
--max-steps 50000
--max-steps 200000
```

Save behavior:

- Rank 0 writes `$results_dir/last.pt` every `--ckpt-every` optimizer steps.
- Rank 0 also writes `$results_dir/last.pt` at the end of each epoch.
- If `--keep-freq > 0`, completed epochs divisible by `keep_freq` also write
  `$results_dir/epoch_{epoch}.pt`.
- Checkpoints contain the student head, fake score head, optional discriminator,
  all optimizer states, `epoch`, `step`, `self_forcing`, and CLI args.
- Self-forcing checkpoints also contain the updated AR backbone.
- `step` is an optimizer step after gradient accumulation, not a microbatch.

Preview behavior:

- `--preview-every <= 0` disables training-time preview sampling.
- If enabled, rank 0 samples `--preview-num` images every `preview_every`
  optimizer steps.
- Images are written to `$results_dir/previews/step_{step}/` by default, with
  individual PNG files and a `grid.png`.
- `--preview-batch-size <= 0` samples all preview images in one batch.
- Preview sampling does not create `.npz` files and does not run FID/IS.
- Other ranks wait at a barrier while rank 0 writes preview images.

Resume and initialization behavior:

- `--resume /path/to/last.pt` fully resumes the same training stage.
- Full resume restores model weights, optimizer states, `epoch`, and `step`.
- `--resume` requires the checkpoint stage to match the current CLI stage:
  teacher-forcing resumes teacher-forcing, and self-forcing resumes self-forcing.
- `--init-from /path/to/last.pt` loads distillation model weights only and starts
  a fresh run from epoch 0 and step 0.
- Use `--init-from` when starting self-forcing from a teacher-forcing stage-1
  checkpoint.
- `--resume` and `--init-from` are mutually exclusive.
- When resuming inside an epoch, the dataloader skips the already consumed
  microbatches implied by `step`, `epoch`, and `grad_accum_steps`.

Logging behavior:

- Logging happens every `--log-every` optimizer steps.
- Metrics are accumulated locally, then reduced across ranks by sum and count.
- Each metric is averaged by its own count. Generator-only metrics are therefore
  averaged only over generator update steps, not over fake-score-only steps.
- `steps/s` counts optimizer steps per second, not microbatches per second.

## Sampling Ablations

`distill_dmd2/sample_ddp.py` always samples with the autoregressive student path.

### Sampling CFG

Flags:

```bash
--cfg-scale 1.0
--cfg-scale 4.5
--cfg-schedule linear
--cfg-schedule constant
```

Suggested comparisons:

```bash
--cfg-scale 1.0
--cfg-scale 2.0
--cfg-scale 4.5 --cfg-schedule linear
--cfg-scale 4.5 --cfg-schedule constant
```

### Sampling Throughput

Flags:

```bash
--per-proc-batch-size 32
--num-fid-samples 50000
--mixed-precision bf16
--mixed-precision none
```

Suggested comparisons:

```bash
--per-proc-batch-size 16
--per-proc-batch-size 32
--per-proc-batch-size 64
```

### Sampling Seeds and Output

Flags:

```bash
--seed 99
--sample-dir samples
--sample-name ""
--to-npz
--keep-pngs
```

Suggested comparisons:

```bash
--seed 0
--seed 99
--seed 123
```

### Checkpoints for Sampling

Flags:

```bash
--teacher-ckpt /path/to/SphereAR_B.pt
--teacher-no-ema
--distill-ckpt /path/to/last.pt
```

The teacher checkpoint provides the frozen AR trunk and VAE. The distilled
checkpoint provides the one-step student head. Self-forcing checkpoints also
provide an `ar_backbone` state dict, and `distill_dmd2/sample_ddp.py` loads it
automatically.

When a self-forcing run is initialized from a teacher-forcing distillation
checkpoint without `ar_backbone`, the AR backbone remains the one loaded from
`--teacher-ckpt`; the one-step head, fake score head, and discriminator are
loaded from the distillation checkpoint.

## Recommended Minimal Sweep

Start from:

```bash
--prefix-mode teacher_forcing
--teacher-sample-steps 100
--teacher-sample-cfg-scale 4.6
--token-sample-size 256
--cfg-scale 1.0
--disc-type resnet
--gan-weight 3e-3
--dfake-gen-update-ratio 5
```

Then sweep one axis at a time:

```bash
# Prefix mode
--prefix-mode teacher_forcing
--prefix-mode real
--prefix-mode teacher_forcing --self-forcing
--prefix-mode teacher_forcing --self-forcing --no-self-forcing-detach-cache

# Token positions
--token-sample-size 32
--token-sample-size 64
--token-sample-size 128
--token-sample-size -1

# Teacher prefix sampling
--teacher-sample-steps 25
--teacher-sample-steps 50
--teacher-sample-steps 100

# CFG
--teacher-sample-cfg-scale 1.0 --cfg-scale 1.0
--teacher-sample-cfg-scale 4.6 --cfg-scale 1.0
--teacher-sample-cfg-scale 4.6 --cfg-scale 4.5

# GAN strength
--gan-domain image --gan-weight 3e-3
--gan-domain latent_grid --gan-weight 3e-3
--gan-domain latent_token --token-sample-size 64 --gan-weight 3e-3
--gan-weight 1e-3
--gan-weight 3e-3
--gan-weight 1e-2

# Compile ablation
--compile-model
```
