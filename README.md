<div align="center">

# Audio-Mamba (AuM)
## Bidirectional State Space Model for Audio Representation Learning
ArXiv Preprint: [https://arxiv.org/abs/2406.03344](https://arxiv.org/abs/2406.03344)
</div>

⚠️🚧 **This repository is under construction. Stay tuned!** 🚧⚠️

## News
- ``05 June 2024`` ArXiv Preprint released: [https://arxiv.org/abs/2406.03344](https://arxiv.org/abs/2406.03344)
- ``22 April 2024`` OpenReview Preprint released: [https://openreview.net/forum?id=RZu0ZlQIUI](https://openreview.net/forum?id=RZu0ZlQIUI)

## Overview
Transformers have rapidly become the preferred choice for audio classification, surpassing methods based on CNNs. However, Audio Spectrogram Transformers (ASTs) exhibit quadratic scaling due to self-attention. The removal of this quadratic self-attention cost presents an appealing direction. Recently, state space models (SSMs), such as Mamba, have demonstrated potential in language and vision tasks in this regard. In this study, we explore whether reliance on self-attention is necessary for audio classification tasks. By introducing Audio Mamba (AuM), the first self-attention-free, purely SSM-based model for audio classification, we aim to address this question. We evaluate AuM on various audio datasets - comprising six different benchmarks - where it achieves comparable or better performance compared to well-established AST model.

<div align="center">
    <img src="audiomamba.png" alt="Pipeline" style="width: 50%;"/>
</div>

## Model Checkpoints
The model checkpoints are available for the following experiments:

#### Base Scratch
These are the checkpoints for the base models with the variant Fo-Bi (b), trained from scratch.
| Dataset                  | #Params | Performance | Checkpoint |
|--------------------------|:---------:|:-------------:|:------------:|
| Audioset (mAP)           | 92.1M   | 32.74       | [Link](https://drive.google.com/file/d/1QgnyvGYxKd-q6twXf4i05jZA5xFIFs8j/view?usp=drive_link) |
| AS-20K (mAP)             | 92.1M   | 14.05       | [Link](https://drive.google.com/file/d/11cbL_vizFFD7i6RvErSSEi9E3gRRIQWA/view?usp=drive_link) |
| VGGSound (Acc)           | 91.9M   | 42.97       | [Link](https://drive.google.com/file/d/1eAn8WEkfnB5pdon8o3PZMwFBsuuPu2L0/view?usp=drive_link) |
| VoxCeleb (Acc)           | 92.7M   | 33.12       | [Link](https://drive.google.com/file/d/1Y3LboHg1RYLsuoKfOT3u4odF6opJZXlw/view?usp=drive_link) |
| Speech Commands V2 (Acc) | 91.4M   | 94.44       | [Link](https://drive.google.com/file/d/1wLFjYZxvJs2YBvDLYqOxVhMJnPKfhX5Z/view?usp=drive_link) |
| Epic Sounds (Acc)        | 91.7M   | 44.92       | [Link](https://drive.google.com/file/d/1vLX3LjAggNAusW6B17s9uc2OoduvnvJi/view?usp=drive_link) |


#### Small ImageNet
These are the checkpoints for the small models with the variant Bi-Bi (c), initialized with ImageNet pretrained weights.
| Dataset | #Params | Performance | Checkpoint |
|---------|:---------:|:-------------:|:------------:|
| Audioset (mAP) | 25.5M | 39.74 |           [Link](https://drive.google.com/file/d/1z-JdZTy52gW7RzhiTQozn6Ly2W4DOs9b/view?usp=drive_link) |
| AS-20K (mAP) | 25.5M | 29.17 |             [Link](https://drive.google.com/file/d/1XDlZEHe0xQXnOLFh3CJVaS5cmZW_7C-t/view?usp=drive_link) |
| VGGSound (Acc) | 25.5M | 49.61 |           [Link](https://drive.google.com/file/d/11mEtjfHjkGGFjxVHvXIAX60KrBgWwWhQ/view?usp=drive_link) |
| VoxCeleb (Acc) | 25.8M | 41.78 |           [Link](https://drive.google.com/file/d/1NoherLBbOP5eE1iMQ8joas1k0lYwAmd8/view?usp=drive_link) |
| Speech Commands V2 (Acc) | 25.2M | 97.61 | [Link](https://drive.google.com/file/d/1jhUKxzUo2TMHrd1a2vojjv1x9De_HyFe/view?usp=drive_link) |
| Epic Sounds (Acc) | 25.4M | 52.69 |        [Link](https://drive.google.com/file/d/1my_kS9COIHGsx4axx8bapN7RBDvp06cK/view?usp=drive_link) |

#### Base AudioSet
These are the checkpoints for the base models with the variant Fo-Bi (b), initialized with AudioSet pretrained weights.

| Dataset | #Params | Performance | Checkpoint |
|---------|:---------:|:-------------:|:------------:|
| VGGSound (Acc) | 91.9M | 46.78 |           [Link](https://drive.google.com/file/d/1spsJXncpEXHKmIvDcB7ddkcgrzARpEeK/view?usp=drive_link) |
| VoxCeleb (Acc) | 92.7M | 41.82 |           [Link](https://drive.google.com/file/d/1dqWSIKTvA0wqKy-XTXYn-MUourMtHGrQ/view?usp=drive_link) |
| Speech Commands V2 (Acc) | 91.4M | 94.82 | [Link](https://drive.google.com/file/d/1ikkU4COOqeCNCVTn4b7LulNr9p4Efr4M/view?usp=drive_link) |
| Epic Sounds (Acc) | 91.7M | 48.31 |        [Link](https://drive.google.com/file/d/1wsRhPqtHryi3PQz1WPJYkMMOPbmOMXrV/view?usp=drive_link) |
