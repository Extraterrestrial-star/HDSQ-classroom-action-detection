# Installation

This document describes how to set up the environment for **HDSQ**, which is built upon **InternVideo** and integrates components from **VideoMamba** for fine-grained classroom action detection.

---

## 1. Create Conda Environment

We recommend using **Python 3.10**.

```bash
conda create -n ava python=3.10 -y
conda activate ava
```

---

## 2. Install PyTorch

Install PyTorch with CUDA 12.1 support:

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

Recommended version:

- PyTorch 2.1.0
- CUDA 12.1

---

## 3. Install VideoMamba Dependencies

Download and prepare the VideoMamba source code:

```bash
wget -O VideoMamba.zip https://ghproxy.net/https://github.com/OpenGVLab/VideoMamba/archive/refs/heads/main.zip
unzip VideoMamba.zip
mv VideoMamba-main VideoMamba
```

Install `causal-conv1d`:

```bash
cd /VideoMamba/mamba/causal-conv1d
pip install --no-build-isolation -e .
```

Install `mamba`:

```bash
cd /VideoMamba/mamba
rm -rf build/ dist/ *.egg-info
MAMBA_FORCE_BUILD="TRUE" pip install --no-build-isolation -e .
```

---

## 4. Install Common Dependencies

```bash
pip install timm==0.4.12
pip install deepspeed==0.13.1
pip install decord
pip install pandas scipy
pip install tensorboardX
pip install einops
pip install tqdm
pip install ftfy regex
pip install yacs imgaug
pip install opencv-python
```

Notes:

- `deepspeed==0.13.1` is recommended for compatibility with **PyTorch 2.1**
- `opencv-python` may need a version adjustment if NumPy compatibility issues occur

---

## 5. Install PyAV

We recommend installing `av` through conda:

```bash
conda install av -c conda-forge
```

This is usually more stable than `pip install av`.
