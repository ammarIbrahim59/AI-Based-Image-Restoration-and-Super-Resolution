# AI-Based Image Restoration and Super-Resolution (NN proj 2)

This project is a PyTorch notebook that covers two image restoration pipelines:
1. **CNN Autoencoder** for image denoising (CelebA).
2. **SRGAN** for 4× super-resolution (DIV2K) with **VGG19 perceptual loss**.

The full implementation lives in **`ProjectNN2FINAL_fixedDisc.ipynb`**.

## What’s inside
- Denoising dataset loader + autoencoder training and visualization.
- SRGAN generator/discriminator, perceptual loss, and training loop.
- Training curves and qualitative result plots.

## Datasets
The notebook expects Kaggle-style paths by default (edit in the **Hyper-parameters** cell if needed):
- **DIV2K**  
  - `/kaggle/input/datasets/joe1995/div2k-dataset/DIV2K_train_HR`  
  - `/kaggle/input/datasets/joe1995/div2k-dataset/DIV2K_valid_HR`
- **CelebA**  
  - `/kaggle/input/celeba-dataset/img_align_celeba`

## Running the notebook
### Recommended (Kaggle)
1. Enable a **T4 GPU** in the notebook settings.
2. Attach the DIV2K and CelebA datasets.
3. Run the notebook top-to-bottom.

### Local (optional)
Install the dependencies and update dataset paths:
```bash
pip install torch torchvision opencv-python-headless tqdm matplotlib numpy
```

The notebook automatically chooses **CUDA** if available; otherwise it falls back to **CPU**.

