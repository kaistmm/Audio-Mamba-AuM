<div align="center">

# Audio-Mamba (AuM)
## Bidirectional State Space Model for Audio Representation Learning
[[ArXiv]](https://arxiv.org/abs/2406.03344), [[IEEE-SPL]](https://ieeexplore.ieee.org/document/10720871)
</div>

## News
- ``13 October 2024`` AuM is accepted at SPL (Signal Processing Letters): [https://ieeexplore.ieee.org/document/10720871](https://ieeexplore.ieee.org/document/10720871)
- ``26 June 2024`` Code cleanup, enhancements and improvements!
- ``16 June 2024`` Added more details for EPIC-SOUNDS!
- ``11 June 2024`` Training scripts released!
- ``10 June 2024`` Setup guide released!
- ``07 June 2024`` Code released! (Initial release — further setup and cleaning in progress.)
- ``06 June 2024`` Checkpoints released!
- ``05 June 2024`` ArXiv Preprint released: [https://arxiv.org/abs/2406.03344](https://arxiv.org/abs/2406.03344)
- ``22 April 2024`` OpenReview Preprint released: [https://openreview.net/forum?id=RZu0ZlQIUI](https://openreview.net/forum?id=RZu0ZlQIUI)

## Index
- [Overview](#overview)
- [Setting Up the Repository](#setting-up-the-repository)
- [Inference](#inference)
- [Training](#training)
- [Model Checkpoints](#model-checkpoints)
- [Citation](#citation)

## Overview
This repository contains the implementation of Audio-Mamba (AuM), a generic, self-attention-free and purely state space model designed for audio classification. It provides the necessary code for training and evaluating the model across various audio classification benchmarks. AuM is built on the works [AST](https://github.com/YuanGongND/ast) and [ViM](https://github.com/hustvl/Vim), and it utilizes Hugging Face's [Accelerate](https://huggingface.co/docs/accelerate/en/index) library to facilitate efficient multi-GPU training.

<div align="center">
    <img src="AuM.png" alt="Pipeline" style="width: 50%;"/>
</div>

## Setting Up the Repository
Please run the following commands to set up the repository:
### Create a Conda Environment
```bash
conda create -n aum python=3.10.13
conda activate aum
```
### Setting Up CUDA and CuDNN
```bash
conda install nvidia/label/cuda-11.8.0::cuda-nvcc
conda install nvidia/label/cuda-11.8.0::cuda

Try: 
conda install anaconda::cudnn
Else:
conda install -c conda-forge cudnn
```
### Installing PyTorch and Other Dependencies
```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
### Installing Mamba Related Packages
```bash
pip install causal_conv1d==1.1.3.post1 mamba_ssm==1.1.3.post1
```
### Enabling Bidirectional SSM Processing
To integrate the modifications for supporting bidirectional processing, copy the `mamba_ssm` folder to the `site-packages` directory of the Python installation within the Conda environment. This folder is directly borrowed from the [ViM](https://github.com/hustvl/Vim) repository. 
```bash
cp -rf vim-mamba_ssm/mamba_ssm $CONDA_PREFIX/lib/python3.10/site-packages
```

## Inference

### Example Inference
An example notebook for inference is provided in the `examples/inference` directory. The notebook demonstrates a minimal example of how to load a trained model and perform inference on a sample audio file.

### Evaluation Scripts
Each dataset folder within the `exps/` directory includes an example evaluation script for AuM (`aum_eval.sh`).

## Training

### Overview
Each dataset's training scripts and relevant files are located within their respective folders under the `exps/` directory. These folders include:

- **Bash files**: Represent configurations for various experiments conducted with the dataset.
- **Data folder** (except for EPIC-SOUNDS): May include example train/test splits, dataset label information, and data sampling weights.

### Executing Training Scripts
To execute the training scripts:

1. **Navigate** to the dataset's directory (e.g., `exps/vggsound/`).
2. **Run** the corresponding script (e.g., `bash aum-base_scratch-vggsound.sh`).

> **Note**: The scripts are prepared for execution but require modification of paths (such as experiment directories) to fit your specific setup.

### Multiple GPU Training
For training on multiple GPUs:

1. **Set GPU IDs**: List the GPU IDs in the `CUDA_VISIBLE_DEVICES` environment variable (e.g., `CUDA_VISIBLE_DEVICES=0,1,2,...`).
2. **Adjust Batch Size**: Set the `batch_size` argument in the script to the desired batch size per GPU.

> **Note**: To maintain the effective batch size from single GPU training, divide the batch size by the number of GPUs.

### EPIC-SOUNDS Dataset
The EPIC-SOUNDS dataset has a distinct training structure:

- **No data folder**: The `epic-sounds/` directory contains only training scripts.
- **Config File**: Additional setup details are available in the `config_default.yaml` file located at `src/epic_sounds/epic_data/` directory. This file includes paths to the dataset folder, training splits, and other relevant default settings (please modify several of these variables according to your setup). Inside `run.py` file, some of the variables from this config file are overriden by the command line arguments. 

For the full reference regarding this dataset, please refer to the [EPIC-SOUNDS repository](https://github.com/epic-kitchens/epic-sounds-annotations). 

## Model Checkpoints
The model checkpoints are available for the following experiments:

### Base Scratch
These are the checkpoints for the base models with the variant `Fo-Bi (b)`, trained from scratch.
| Dataset                  | #Params | Performance | Checkpoint |
|--------------------------|:---------:|:-------------:|:------------:|
| Audioset (mAP)           | 92.1M   | 32.74       | [Link](https://drive.google.com/file/d/1QgnyvGYxKd-q6twXf4i05jZA5xFIFs8j/view?usp=drive_link) |
| AS-20K (mAP)             | 92.1M   | 14.05       | [Link](https://drive.google.com/file/d/11cbL_vizFFD7i6RvErSSEi9E3gRRIQWA/view?usp=drive_link) |
| VGGSound (Acc)           | 91.9M   | 42.97       | [Link](https://drive.google.com/file/d/1eAn8WEkfnB5pdon8o3PZMwFBsuuPu2L0/view?usp=drive_link) |
| VoxCeleb (Acc)           | 92.7M   | 33.12       | [Link](https://drive.google.com/file/d/1Y3LboHg1RYLsuoKfOT3u4odF6opJZXlw/view?usp=drive_link) |
| Speech Commands V2 (Acc) | 91.4M   | 94.44       | [Link](https://drive.google.com/file/d/1wLFjYZxvJs2YBvDLYqOxVhMJnPKfhX5Z/view?usp=drive_link) |
| Epic Sounds (Acc)        | 91.7M   | 44.92       | [Link](https://drive.google.com/file/d/1vLX3LjAggNAusW6B17s9uc2OoduvnvJi/view?usp=drive_link) |


### Small ImageNet
These are the checkpoints for the small models with the variant `Bi-Bi (c)`, initialized with ImageNet pretrained weights.
| Dataset | #Params | Performance | Checkpoint |
|---------|:---------:|:-------------:|:------------:|
| Audioset (mAP) | 25.5M | 39.74 |           [Link](https://drive.google.com/file/d/1z-JdZTy52gW7RzhiTQozn6Ly2W4DOs9b/view?usp=drive_link) |
| AS-20K (mAP) | 25.5M | 29.17 |             [Link](https://drive.google.com/file/d/1XDlZEHe0xQXnOLFh3CJVaS5cmZW_7C-t/view?usp=drive_link) |
| VGGSound (Acc) | 25.5M | 49.61 |           [Link](https://drive.google.com/file/d/11mEtjfHjkGGFjxVHvXIAX60KrBgWwWhQ/view?usp=drive_link) |
| VoxCeleb (Acc) | 25.8M | 41.78 |           [Link](https://drive.google.com/file/d/1NoherLBbOP5eE1iMQ8joas1k0lYwAmd8/view?usp=drive_link) |
| Speech Commands V2 (Acc) | 25.2M | 97.61 | [Link](https://drive.google.com/file/d/1jhUKxzUo2TMHrd1a2vojjv1x9De_HyFe/view?usp=drive_link) |
| Epic Sounds (Acc) | 25.4M | 53.45 |        [Link](https://drive.google.com/file/d/1i9ANh01FWB8UY9ruQ81Ov5UnoXuhq0PQ/view?usp=drive_link) |

### Base AudioSet
These are the checkpoints for the base models with the variant `Fo-Bi (b)`, initialized with AudioSet pretrained weights.

| Dataset | #Params | Performance | Checkpoint |
|---------|:---------:|:-------------:|:------------:|
| VGGSound (Acc) | 91.9M | 46.78 |           [Link](https://drive.google.com/file/d/1spsJXncpEXHKmIvDcB7ddkcgrzARpEeK/view?usp=drive_link) |
| VoxCeleb (Acc) | 92.7M | 41.82 |           [Link](https://drive.google.com/file/d/1dqWSIKTvA0wqKy-XTXYn-MUourMtHGrQ/view?usp=drive_link) |
| Speech Commands V2 (Acc) | 91.4M | 94.82 | [Link](https://drive.google.com/file/d/1ikkU4COOqeCNCVTn4b7LulNr9p4Efr4M/view?usp=drive_link) |
| Epic Sounds (Acc) | 91.7M | 48.31 |        [Link](https://drive.google.com/file/d/1wsRhPqtHryi3PQz1WPJYkMMOPbmOMXrV/view?usp=drive_link) |

## Citation
If you find this work useful, please consider citing us:

```bibtex
@article{erol2024audio,
  author={Erol, Mehmet Hamza and Senocak, Arda and Feng, Jiu and Chung, Joon Son},
  journal={IEEE Signal Processing Letters}, 
  title={Audio Mamba: Bidirectional State Space Model for Audio Representation Learning}, 
  year={2024},
  volume={31},
  pages={2975-2979},
  doi={10.1109/LSP.2024.3483009}
}
```
