# AI-Based Image Restoration and Super-Resolution

Two image-restoration pipelines built and trained from scratch in PyTorch: a CNN
autoencoder for **denoising**, and an **SRGAN** for 4× super-resolution with VGG19
perceptual loss.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Results

| Model | Params | Metric | Result |
|---|---|---|---|
| SRGAN generator (SRResNet) | 1.55 M | Validation PSNR | 10.49 dB → **22.15 dB** |
| SRGAN discriminator (PatchGAN) | 23.6 M | — | Converges to near-zero loss |
| Denoising autoencoder | 1.93 M | Validation MSE | Not run — see note below |

SRGAN trained for 50 epochs on DIV2K (800 train / 100 validation images), peaking at
**22.15 dB** validation PSNR at epoch 40.

> **Note on the denoising results.** In the recorded run the CelebA dataset was not
> attached to the notebook, so Part 1 reported *"CelebA not found — skipping"* and its
> training cells have no stored output. The dataset lookup has since been fixed (the
> Kaggle CelebA mirror nests images one level deeper than the folder name implies, and
> path resolution is now recursive) — attach the dataset and re-run to populate that
> section. Part 2 ran fully and has complete stored results.

## Architecture

### Part 1 — Denoising autoencoder

Symmetric encoder–decoder with **skip connections**. Three encoder stages
(32 → 64 → 128 channels) each halve spatial resolution via max pooling, a 256-channel
bottleneck sits at the centre, and three decoder stages upsample with transposed
convolutions. Each decoder stage concatenates the matching encoder feature map before
convolving — the skips preserve high-frequency facial detail that a plain bottleneck
would blur away. Trained with MSE against Gaussian noise at σ = 0.1.

### Part 2 — SRGAN

**Generator (SRResNet):** 16 residual blocks at 64 channels, then two ×2 sub-pixel
convolution (PixelShuffle) upsampling blocks for ×4 total, with a long skip connection
past the residual trunk.

**Discriminator (PatchGAN):** eight strided convolution blocks alternating feature growth
and downsampling, then adaptive average pooling so any HR crop size works without
changing the classifier head.

**Loss:**

| Component | Weight | Purpose |
|---|---|---|
| VGG19 perceptual (`relu3_4`) | 1.0 | Match deep features, not pixels — drives perceptual sharpness |
| Pixel MSE | 1.0 | Anchors colour and structure, stabilises early training |
| Adversarial (BCE) | 5e-3 | Pushes outputs toward the natural-image manifold |

Pixel-wise MSE alone drives a generator toward the blurry mean of all plausible
reconstructions. Matching VGG19 activations instead rewards realistic texture — this is
the single biggest quality factor in the pipeline.

## Fixing GAN mode collapse

The first SRGAN attempt collapsed: the discriminator won outright within a few epochs,
the adversarial gradient vanished, and the generator stopped improving. Five changes
produced a converging run:

1. **Asymmetric learning rates** — discriminator at 2e-5 vs generator at 1e-4, so the
   discriminator cannot outpace the generator.
2. **One-sided label smoothing** — real labels at 0.9 rather than 1.0, preventing
   overconfidence on real images. The generator still targets a genuine 1.0 so its own
   adversarial gradient is not capped.
3. **Instance noise** — Gaussian noise added to both real and fake discriminator inputs,
   decayed linearly to zero across training, keeping the discriminator's task hard early
   so gradients keep flowing.
4. **Explicit `n_disc_steps`** — the discriminator/generator update ratio is a tunable
   parameter rather than an implicit 1:1.
5. **Stronger adversarial weight** — `LAMBDA_ADV` raised from 1e-3 to 5e-3.

Both optimisers use cosine-annealed learning rates over the 50 epochs.

**Honest limitation:** even with these fixes the discriminator loss frequently collapses
to ~0.000, so the generator improves mostly through the perceptual and pixel losses. The
run behaves closer to an SRResNet with a weak adversarial regulariser than to a fully
balanced SRGAN.

## Datasets

| Dataset | Used for | Default path |
|---|---|---|
| [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) | Super-resolution | `/kaggle/input/datasets/joe1995/div2k-dataset/DIV2K_{train,valid}_HR` |
| [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | Denoising | `/kaggle/input/celeba-dataset/...` (probed recursively) |

Image discovery is recursive and path candidates are probed in order, so differently
nested dataset mirrors resolve without editing the notebook.

## Running it

### Kaggle (recommended)

1. Select a **T4 GPU** accelerator. The older P100 (sm_60) is not supported by current
   PyTorch builds.
2. Attach the DIV2K and CelebA datasets.
3. Run the install cell, **restart the kernel**, then run the remaining cells in order.

### Local

```bash
pip install -r requirements.txt
jupyter notebook image_restoration_srgan.ipynb
```

Update the dataset paths in the hyper-parameters cell. The notebook uses CUDA when
available and falls back to CPU otherwise — though CPU training is impractically slow
for the SRGAN.

## Hyper-parameters

All tunables live in one cell:

| Setting | Default | Purpose |
|---|---|---|
| `BATCH_SIZE` | `16` | Batch size for both pipelines |
| `DENOISE_EPOCHS` / `DENOISE_LR` | `20` / `1e-3` | Autoencoder training |
| `NOISE_STD` | `0.1` | Gaussian noise σ added to inputs |
| `SR_EPOCHS` | `50` | SRGAN training length |
| `SR_LR` / `SR_LR_D` | `1e-4` / `2e-5` | Generator / discriminator learning rates |
| `UPSCALE_FACTOR` | `4` | Super-resolution factor |
| `HR_CROP_SIZE` | `128` | HR patch size (LR patch = 128/4 = 32) |
| `LAMBDA_CONTENT` / `LAMBDA_ADV` | `1.0` / `5e-3` | Perceptual / adversarial loss weights |
| `SEED` | `42` | Reproducibility |

## Outputs

- `srgan_generator.pth`, `srgan_discriminator.pth` — SRGAN weights
- `denoising_autoencoder.pth` — written only if Part 1 actually trained

## Limitations & next steps

- The discriminator still overpowers the generator (see above); a relativistic or hinge
  adversarial loss would help.
- Only PSNR is reported. PSNR systematically favours blurry output, so **SSIM and LPIPS**
  should be measured alongside it.
- Denoising is evaluated on 64×64 CelebA crops — small compared to real-world images.
- 50 epochs on 800 images is short for a GAN; longer training on more data would improve
  perceptual quality.

## License

[MIT](LICENSE)
