# SphereAR Consistency Distillation

This folder implements a SphereAR-specific consistency distillation path using
the OpenAI `consistency_models` repository as the algorithm reference.  It is
kept separate from the original SphereAR training code.

## Scope

Implemented:

- Teacher-model consistency distillation for the SphereAR diffusion head.
- Frozen SphereAR teacher for VAE, AR conditioning, and teacher ODE steps.
- Student and EMA target consistency heads initialized from the teacher
  diffusion head.
- Only provides 'real' prefix mode.
- L1/L2/pseudo-Huber latent-token consistency losses.
- Autoregressive one-step or multi-step consistency sampling.

Not implemented yet:

- Consistency training from scratch without a teacher model.
- Self-forcing stage that trains the AR backbone.
- LPIPS loss over decoded images.

## Algorithm

SphereAR's diffusion head is trained with the interpolation

```text
x_t = (1 - t) noise + t x0
```

OpenAI consistency models use additive noise

```text
x_sigma = x0 + sigma noise
```

The implementation maps `sigma` to the SphereAR diffusion time with
`t = 1 / (1 + sigma)`, so `x_t = t * x_sigma`.  This lets the pretrained
SphereAR diffusion head act as the teacher score/denoiser without changing the
SphereAR model code.

For each selected raster token:

1. Sample adjacent Karras noise levels `sigma_i > sigma_{i+1}`.
2. Add noise to a latent token.
3. Predict clean latents with the trainable consistency head at `sigma_i`.
4. Use the frozen teacher diffusion head and a Heun step to move the noisy
   token to `sigma_{i+1}`.
5. Predict the target clean latent with the EMA target head at `sigma_{i+1}`.
6. Minimize a weighted consistency loss between the student and target outputs.

The target head is updated after every optimizer step with `--target-ema`.

## Entry Points

Training:

```bash
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 \
  distill_cm/train.py \
  --teacher-ckpt $TEACHER_CKPT \
  --data-path $DATA_PATH \
  --results-dir $RESULT_DIR \
  --model SphereAR-B \
  --image-size 256 \
  --patch-size 16 \
  --latent-dim 16 \
  --global-batch-size 512 \
  --grad-accum-steps 1 \
  --num-scales 40 \
  --target-ema 0.95 \
  --token-sample-size 256 \
  --loss-norm l2 \
  --weight-schedule uniform \
  --mixed-precision bf16
```

Sampling:

```bash
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 \
  distill_cm/sample_ddp.py \
  --teacher-ckpt $TEACHER_CKPT \
  --distill-ckpt $RESULT_DIR/last.pt \
  --model SphereAR-B \
  --image-size 256 \
  --patch-size 16 \
  --latent-dim 16 \
  --num-consistency-steps 1 \
  --num-fid-samples 50000 \
  --to-npz
```

Use `--head target` for the EMA target head, which is the default.  Multi-step
consistency sampling can use either `--num-consistency-steps N` or explicit
descending sigmas via `--sampling-sigmas 80,10,1`.
