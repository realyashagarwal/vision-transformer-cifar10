# Vision Transformer from Scratch on CIFAR-10

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A research-focused implementation of Vision Transformer (ViT) trained from scratch on CIFAR-10, achieving **90.78% test accuracy** without large-scale pre-training.

**Repository**: [github.com/realyashagarwal/vision-transformer-cifar10](https://github.com/realyashagarwal/vision-transformer-cifar10)

---

## Abstract

This work presents a comprehensive implementation of Vision Transformers (ViT) trained from scratch on the CIFAR-10 dataset. We demonstrate that through careful integration of modern architectural improvements and training methodologies, Vision Transformers can achieve competitive performance (90.78% test accuracy) on small-scale datasets without requiring pre-training on large corpora such as ImageNet-21k or JFT-300M. Our implementation incorporates LayerScale for training stability, stochastic depth for regularization, and advanced data augmentation strategies including MixUp and CutMix.

---

## Table of Contents

- [Introduction](#introduction)
- [Methodology](#methodology)
  - [Architecture](#architecture)
  - [Training Techniques](#training-techniques)
  - [Hyperparameters](#hyperparameters)
- [Results](#results)
  - [Quantitative Results](#quantitative-results)
  - [Per-Class Analysis](#per-class-analysis)
- [Experimental Setup](#experimental-setup)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)
- [Citation](#citation)
- [License](#license)

---

## Introduction

Vision Transformers (ViT) [1] have demonstrated remarkable performance on image recognition tasks when pre-trained on large-scale datasets. However, training ViTs from scratch on smaller datasets has proven challenging due to their lack of inductive biases inherent to convolutional architectures. This project addresses these challenges by incorporating several state-of-the-art techniques that enable successful training on CIFAR-10 without pre-training.

### Key Contributions

1. Implementation of ViT with modern architectural improvements (LayerScale, Pre-Normalization)
2. Integration of advanced regularization techniques (Stochastic Depth, Label Smoothing)
3. Application of strong data augmentation strategies (MixUp, CutMix)
4. Demonstration of 90.78% test accuracy on CIFAR-10 without pre-training

---

## Methodology

### Architecture

Our implementation is based on the Vision Transformer architecture [1] with the following specifications:

| Component | Configuration |
|-----------|---------------|
| Input Resolution | 32 × 32 × 3 |
| Patch Size | 4 × 4 |
| Number of Patches | 64 |
| Embedding Dimension | 384 |
| Transformer Depth | 12 blocks |
| Attention Heads | 6 |
| MLP Expansion Ratio | 4× |
| Total Parameters | 21,423,562 |

#### Architectural Enhancements

**1. LayerScale** [2]  
Introduced learnable scaling parameters for each residual branch to improve training stability in deep transformers.

**2. Pre-Normalization**  
LayerNorm is applied before attention and MLP blocks, enabling smoother gradient flow compared to post-normalization.

**3. Stochastic Depth (DropPath)** [3]  
Randomly drops entire residual blocks during training with linearly increasing probability from 0.0 to 0.1 across layers.

### Training Techniques

#### Optimization

**AdamW Optimizer** [4]  
We employ the AdamW optimizer with decoupled weight decay:
- Learning Rate: 1e-3
- Weight Decay: 0.05
- β₁ = 0.9, β₂ = 0.999
- Batch Size: 128

**Learning Rate Schedule**  
A warmup cosine schedule is applied:
- Linear warmup: 20 epochs (0 → 1e-3)
- Cosine decay: 180 epochs (1e-3 → 1e-5)

#### Regularization

**Label Smoothing** [5]  
Cross-entropy loss with label smoothing (ε = 0.1) to prevent overconfidence.

**Stochastic Depth**  
Drop probability increases linearly from 0.0 (first layer) to 0.1 (final layer).

#### Data Augmentation

**MixUp** [6]  
Linear interpolation of image pairs with α = 0.2:
```
x̃ = λxᵢ + (1-λ)xⱼ
ỹ = λyᵢ + (1-λ)yⱼ
```

**CutMix** [7]  
Random patch replacement between images with α = 1.0, promoting localization ability.

**Standard Augmentations**
- Random horizontal flip
- Random crop with padding
- Color jitter

### Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Training Epochs | 200 |
| Batch Size | 128 |
| Base Learning Rate | 1e-3 |
| Warmup Epochs | 20 |
| Weight Decay | 0.05 |
| Label Smoothing | 0.1 |
| MixUp α | 0.2 |
| CutMix α | 1.0 |
| DropPath Rate | 0.0 → 0.1 |

---

## Results

### Quantitative Results

Our model achieves **90.78%** test accuracy on CIFAR-10, demonstrating that Vision Transformers can be effectively trained from scratch on small-scale datasets when equipped with appropriate training techniques.

```
==================================================
VISION TRANSFORMER - CIFAR-10 RESULTS
==================================================
Model Configuration:
  - Patch Size: 4x4
  - Embedding Dim: 384
  - Depth: 12 blocks
  - Heads: 6
  - Parameters: 21,423,562

Training Configuration:
  - Epochs: 200
  - Batch Size: 128
  - Learning Rate: 0.001 (with warmup)
  - Augmentations: CutMix + MixUp + RandAugment

Results:
  - Best Test Accuracy: 90.78%
  - Overall Test Accuracy: 90.78%
==================================================
```

### Per-Class Analysis

The model demonstrates strong performance across all CIFAR-10 classes, with particularly high accuracy on vehicle categories.

| Class | Accuracy | Class | Accuracy |
|-------|----------|-------|----------|
| car | 95.80% | ship | 95.10% |
| truck | 93.10% | horse | 93.00% |
| frog | 92.80% | plane | 92.50% |
| bird | 89.90% | deer | 88.80% |
| dog | 86.00% | cat | 80.80% |

**Observations:**
- Highest accuracy on rigid, structured objects (vehicles: car, ship, truck)
- Lower accuracy on natural categories with high intra-class variation (cat, dog)
- Average performance gap between best and worst class: 15.0%

---

## Experimental Setup

### Software Environment

```
Python 3.10
PyTorch 2.0+
torchvision
numpy
matplotlib
jupyter
```

### Computational Resources

Training was conducted on consumer-grade hardware to demonstrate accessibility of the approach.

### Reproducibility

All hyperparameters, random seeds, and training configurations are documented in the codebase. The pre-trained checkpoint is provided for result verification.

---

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ VRAM (for batch size 128)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/realyashagarwal/vision-transformer-cifar10.git
cd vision-transformer-cifar10
```

2. **Create a virtual environment**

**Option A: Conda**
```bash
conda create -n vit_project python=3.10
conda activate vit_project
```

**Option B: venv**
```bash
python -m venv vit_env
source vit_env/bin/activate  # Windows: vit_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Usage

### Evaluation

To reproduce the reported results using the pre-trained checkpoint:

```bash
jupyter notebook notebooks/vit_cifar10_analysis.ipynb
```

Select the `vit_project` kernel and execute the evaluation cells. The notebook will:
1. Load the CIFAR-10 test set
2. Initialize the model architecture
3. Load pre-trained weights from `checkpoints/best_vit_cifar10.pth`
4. Evaluate on the test set and display per-class accuracies

**Expected Output:**
```
Loading test data...
100%|██████████| 170M/170M [11:04<00:00, 256kB/s]  
Loading checkpoint from ../checkpoints/best_vit_cifar10.pth...
Evaluating model on test set...

Overall Test Accuracy: 90.78%
Per-class Accuracy:
------------------------------
plane     : 92.50%
car       : 95.80%
bird      : 89.90%
cat       : 80.80%
deer      : 88.80%
dog       : 86.00%
frog      : 92.80%
horse     : 93.00%
ship      : 95.10%
truck     : 93.10%
```

### Training

The training loop is commented out in the notebook but can be enabled by uncommenting the relevant cells. Training configuration is specified in `src/utils.py` and `src/engine.py`.

---

## References

[1] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). *International Conference on Learning Representations (ICLR)*.

[2] Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jégou, H. (2021). [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239). *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 32-42.

[3] Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K. Q. (2016). [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382). *European Conference on Computer Vision (ECCV)*, 646-661.

[4] Loshchilov, I., & Hutter, F. (2019). [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101). *International Conference on Learning Representations (ICLR)*.

[5] Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567). *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2818-2826.

[6] Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412). *International Conference on Learning Representations (ICLR)*.

[7] Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., & Yoo, Y. (2019). [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/abs/1905.04899). *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 6023-6032.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{agarwal2025vit_cifar10,
  author = {Agarwal, Yash},
  title = {Vision Transformer from Scratch on CIFAR-10},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/realyashagarwal/vision-transformer-cifar10}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

**Acknowledgments**: We thank the PyTorch team for their excellent framework and the authors of the original Vision Transformer paper for their foundational work.