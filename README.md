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
