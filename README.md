# Two-Stage Super-Resolution GAN (SRGAN)

## Overview

This project implements a custom **Two-Stage Super-Resolution (SR)** pipeline designed to reconstruct high-fidelity images from blurred, low-resolution inputs. Unlike standard upsampling methods, this architecture leverages a **Generative Adversarial Network (GAN)** approach combined with **perceptual feature matching**.

The model integrates **RRDBNet** (Residual-in-Residual Dense Block Network) and **PixelShuffle** upscaling to balance structural accuracy with perceptual realism, effectively eliminating checkerboard artifacts while recovering fine-grained details.

## Architecture

The pipeline operates in two distinct stages to ensure stable feature extraction and high-quality upscaling.

### Stage 1: Feature Reconstruction
* **Backbone:** `RRDBNet`
* **Core Components:** Multiple **Dense Residual Blocks** are utilized for deep feature extraction.
* **Mechanism:** Uses **Residual-in-Residual** connections to stabilize gradients during deep network training, allowing the model to learn complex low-frequency structures.

### Stage 2: Super-Resolution Upscaling
* **Upsampling Method:** `PixelShuffle` (Sub-pixel Convolution).
* **Goal:** Efficient spatial resolution increase ($H \times W \to sH \times sW$).
* **Benefit:** Preserves fine-grained spatial details and prevents the checkerboard artifacts often seen with Transposed Convolutions.

## Loss Functions & Optimization

To achieve photorealistic results, the Generator is optimized using a composite objective function:

$$\mathcal{L}_{total} = \lambda_{pix}\mathcal{L}_{pixel} + \lambda_{perc}\mathcal{L}_{perc} + \lambda_{adv}\mathcal{L}_{adv}$$

### 1. Pixel Loss ($\mathcal{L}_{pixel}$)
We use L1 loss for low-level pixel accuracy to ensure the generated image color space remains consistent with the ground truth.

$$\mathcal{L}_{pixel} = \| I_{SR} - I_{HR} \|_1$$

### 2. Perceptual Loss ($\mathcal{L}_{perc}$)
Computed using feature maps extracted from a pre-trained **VGG16** network. This optimizes for structural similarity rather than just pixel-perfect matches.

$$\mathcal{L}_{perc} = \| \phi(x_{SR}) - \phi(x_{HR}) \|_2$$

* $\phi(\cdot)$: Feature maps from the VGG16 layer.
* $x_{SR}$: Super-Resolved (Generated) Image.
* $x_{HR}$: High-Resolution (Ground Truth) Image.

### 3. Adversarial Loss ($\mathcal{L}_{adv}$)
The GAN component forces the generator to create textures that are indistinguishable from real images by the discriminator.

$$\mathcal{L}_{adv} = -\log(D(G(I_{LR})))$$

## Adversarial Training Strategy

* **Discriminator:** Trained to distinguish between the Super-Resolved images ($x_{SR}$) and Ground Truth images ($x_{HR}$).
* **Generator:** Trained simultaneously to minimize reconstruction error (Perceptual/Pixel loss) and fool the Discriminator (Adversarial loss).

##  Results & Performance

The model was evaluated on a held-out test set using resolution-preserving resizing.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **PSNR** | **31–32 dB** | Peak Signal-to-Noise Ratio indicating high structural fidelity. |
| **SSIM** | **~0.95** | Structural Similarity Index confirming strong detail reconstruction. |

*Note: The model demonstrates improved perceptual sharpness compared to standard MSE-based optimization methods.*

##  Tech Stack

* **Language:** Python
* **Deep Learning Framework:** PyTorch (Implementation includes `torch.nn`, `torch.optim`)
* **Feature Extraction:** VGG16 (Pre-trained on ImageNet)
* **Acceleration:** CUDA + Automatic Mixed Precision (AMP)

## Getting Started

### Prerequisites
```bash
pip install torch torchvision numpy opencv-python matplotlib
