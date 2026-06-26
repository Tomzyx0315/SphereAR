# SphereAR DMD2 One-Step Distillation

This repository adds a DMD2-style one-step distillation pipeline for the SphereAR diffusion head.

## Algorithm

The distillation pipeline trains a one-step student head on top of a frozen SphereAR teacher. The teacher provides the VAE, autoregressive trunk, class conditioning path, and pretrained diffusion head. The default prefix mode is teacher forcing: the teacher first samples a clean latent token sequence with its multi-step diffusion head, then the frozen AR trunk builds per-token conditions from the teacher clean prefix. The student maps one noise vector plus the current AR condition to one hyperspherical latent token.

The student head is initialized from the teacher diffusion head. For each generated latent token, the one-step prediction is normalized with the same VAE latent normalization used by SphereAR sampling. The generated latent grid is decoded by the frozen VAE for the GAN generator loss.

The training code also supports `real` prefixes, which use VAE latents from the current real batch. Sampling uses the autoregressive student path.

The generator objective is the weighted sum of:

- Distribution matching loss: generated latents are noised at a random timestep, then scored by the frozen teacher diffusion head and a trainable fake score head. Their predicted clean-latent gap defines the DMD gradient surrogate used to update the one-step student.
- GAN generator loss: generated samples are evaluated by a class-conditional discriminator. The default domain is decoded images; `--gan-domain latent_grid`, `--gan-domain latent_token`, and `--gan-domain none` are available ablations.

The fake score head is also initialized from the teacher diffusion head and is trained online on generated latents. The discriminator is trained on real ImageNet images with their dataset labels and decoded student samples with their sampled class labels. The same generated batch is reused for the student update, fake score update, and discriminator fake branch.

Classifier-free guidance has two controls. `--teacher-sample-cfg-scale` controls the teacher samples used as clean prefixes in `teacher_forcing` mode. `--cfg-scale` controls conditional/null score combination in the DMD loss and the student token prediction before latent normalization. The single-node baseline uses guided teacher prefixes with `--teacher-sample-cfg-scale 2.5` and keeps distillation CFG at `--cfg-scale 1.0`.

`--token-sample-size` subsamples raster positions for the distribution matching and fake score losses. It also controls the sampled positions for `--gan-domain latent_token`; image and latent-grid GANs still use the full generated grid.

Implemented entry points:

- `distill_dmd2/train.py`: DDP training with `torchrun`
- `distill_dmd2/sample_ddp.py`: DDP sampling with `torchrun`
- `distill_dmd2/distiller.py`: teacher-forced prefixes, real prefixes, CFG, distribution matching, fake score training
- `distill_dmd2/heads.py`: one-step student head and fake score head
- `distill_dmd2/gan.py`: projection ResNet discriminator, discriminator builders, and adversarial losses

## Environment

Use the same CUDA/PyTorch/FlashAttention environment as SphereAR training. The code expects CUDA and NCCL for training and sampling.

Example package versions:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import flash_attn; print(flash_attn.__version__)"
```

## Data Preparation

Prepare ImageNet training data as either the official training tar or an ImageFolder directory.

Tar layout:

```bash
export DATA_PATH=/path/to/ILSVRC2012_img_train.tar
```

Folder layout:

```bash
export DATA_PATH=/path/to/imagenet/train
```

When using the tar file, the dataloader creates an index file next to it:

```bash
/path/to/ILSVRC2012_img_train.tar.index
```

Make sure the directory containing the tar file is writable, or place the tar on local node storage before launching training.

For FID/IS evaluation, download the ImageNet 256 reference batch:

```bash
export REF_NPZ=/path/to/VIRTUAL_imagenet256_labeled.npz
wget -O $REF_NPZ \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
```

## Checkpoint Preparation

Prepare a pretrained SphereAR checkpoint containing either `ema` or `model` weights. The checkpoint is expected to include the VAE, AR trunk, and diffusion head; the distillation scripts do not load a separate VAE checkpoint.

Example with a local checkpoint:

```bash
export TEACHER_CKPT=/path/to/SphereAR_B.pt
```

Example download location:

- `SphereAR_B.pt`: https://huggingface.co/guolinke/SphereAR/blob/main/SphereAR_B.pt
- `SphereAR_L.pt`: https://huggingface.co/guolinke/SphereAR/blob/main/SphereAR_L.pt
- `SphereAR_H.pt`: https://huggingface.co/guolinke/SphereAR/blob/main/SphereAR_H.pt

Use the matching model name when launching distillation:

```bash
export MODEL=SphereAR-B
```

## Training

Single-node 8-GPU A100 baseline example:

```bash
export DATA_PATH=/path/to/ILSVRC2012_img_train.tar
export TEACHER_CKPT=/path/to/SphereAR_B.pt
export RESULT_DIR=/path/to/runs/spherear_b_dmd2
export MODEL=SphereAR-B

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 \
  distill_dmd2/train.py \
  --teacher-ckpt $TEACHER_CKPT \
  --data-path $DATA_PATH \
  --results-dir $RESULT_DIR \
  --model $MODEL \
  --image-size 256 \
  --patch-size 16 \
  --latent-dim 16 \
  --global-batch-size 512 \
  --grad-accum-steps 4 \
  --student-lr 2e-6 \
  --fake-score-lr 2e-6 \
  --disc-lr 2e-6 \
  --dm-weight 1.0 \
  --gan-weight 3e-3 \
  --dfake-gen-update-ratio 5 \
  --prefix-mode teacher_forcing \
  --teacher-sample-steps 100 \
  --teacher-sample-cfg-scale 2.5 \
  --token-sample-size 256 \
  --cfg-scale 1.0 \
  --cfg-schedule linear \
  --gan-domain image \
  --disc-type resnet \
  --disc-dim 64 \
  --gan-loss hinge \
  --log-every 50 \
  --ckpt-every 1000 \
  --preview-every 1000 \
  --preview-num 16 \
  --preview-batch-size 8 \
  --mixed-precision bf16
```

Multi-node example:

```bash
export DATA_PATH=/path/to/ILSVRC2012_img_train.tar
export TEACHER_CKPT=/path/to/SphereAR_B.pt
export RESULT_DIR=/path/to/runs/spherear_b_dmd2
export MODEL=SphereAR-B
export WORKER_NUM=2
export NODE_RANK=0
export WORKER_0_HOST=master.host
export WORKER_0_PORT=29500

torchrun --nproc_per_node=8 \
  --nnodes=$WORKER_NUM \
  --node_rank=$NODE_RANK \
  --master_addr=$WORKER_0_HOST \
  --master_port=$WORKER_0_PORT \
  distill_dmd2/train.py \
  --teacher-ckpt $TEACHER_CKPT \
  --data-path $DATA_PATH \
  --results-dir $RESULT_DIR \
  --model $MODEL \
  --image-size 256 \
  --patch-size 16 \
  --latent-dim 16 \
  --global-batch-size 128 \
  --grad-accum-steps 1 \
  --student-lr 2e-6 \
  --fake-score-lr 2e-6 \
  --disc-lr 2e-6 \
  --dm-weight 1.0 \
  --gan-weight 3e-3 \
  --dfake-gen-update-ratio 5 \
  --prefix-mode teacher_forcing \
  --teacher-sample-steps 100 \
  --teacher-sample-cfg-scale 1.0 \
  --token-sample-size 64 \
  --cfg-scale 1.0 \
  --gan-domain image \
  --disc-type resnet \
  --gan-loss hinge \
  --mixed-precision bf16
```

Training writes checkpoints to:

```bash
$RESULT_DIR/last.pt
$RESULT_DIR/epoch_*.pt
```

The single-node command logs every 50 optimizer steps, overwrites `last.pt` every 1000 optimizer steps, and saves 16 preview PNGs plus a grid every 1000 optimizer steps. Training-time preview sampling does not run FID/IS.

Resume from an existing run:

```bash
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 \
  distill_dmd2/train.py \
  --teacher-ckpt $TEACHER_CKPT \
  --data-path $DATA_PATH \
  --results-dir $RESULT_DIR \
  --resume $RESULT_DIR/last.pt \
  --model $MODEL \
  --image-size 256 \
  --patch-size 16 \
  --latent-dim 16 \
  --global-batch-size 64 \
  --grad-accum-steps 1 \
  --prefix-mode teacher_forcing \
  --teacher-sample-steps 100
```

Useful training options:

```bash
--max-steps 200000
--grad-accum-steps 2
--ckpt-every 1000
--log-every 50
--preview-every 1000 --preview-num 16
--disc-dim 64
--prefix-mode real
--gan-domain image
--gan-domain latent_grid
--gan-domain latent_token
--gan-domain none
--token-sample-size 64
--token-sample-size -1
--teacher-sample-cfg-scale 1.0
--cfg-scale 1.0
--cfg-scale 4.5
```

## Sampling

Sample with the distilled student:

```bash
export TEACHER_CKPT=/path/to/SphereAR_B.pt
export DISTILL_CKPT=/path/to/runs/spherear_b_dmd2/last.pt
export SAMPLE_DIR=/path/to/samples
export MODEL=SphereAR-B

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 \
  distill_dmd2/sample_ddp.py \
  --teacher-ckpt $TEACHER_CKPT \
  --distill-ckpt $DISTILL_CKPT \
  --sample-dir $SAMPLE_DIR \
  --model $MODEL \
  --image-size 256 \
  --patch-size 16 \
  --latent-dim 16 \
  --per-proc-batch-size 32 \
  --num-fid-samples 50000 \
  --cfg-scale 1.0 \
  --cfg-schedule linear \
  --mixed-precision bf16 \
  --to-npz
```

The command writes PNGs and, with `--to-npz`, creates a `.npz` file for evaluation. Add `--keep-pngs` to retain the image folder after the `.npz` file is created.

## Evaluation

Run the evaluator on the generated `.npz`:

```bash
export REF_NPZ=/path/to/VIRTUAL_imagenet256_labeled.npz
export GEN_NPZ=/path/to/generated_samples.npz

python evaluator.py $REF_NPZ $GEN_NPZ
```

The evaluator follows the OpenAI guided-diffusion ImageNet reference batch protocol.
