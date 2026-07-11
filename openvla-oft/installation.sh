#!/usr/bin/env bash
set -e

ENV_NAME=openvla-oft

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda environment $ENV_NAME already exists; reusing it."
else
    conda create -n "$ENV_NAME" python=3.10 -y
fi

conda activate "$ENV_NAME"
conda install conda-forge::imagemagick -y # This is for LIBERO-plus

pip install torch torchvision torchaudio
pip install -e .
pip install packaging ninja

# You need to download the flash_attn wheel file from https://github.com/Dao-AILab/flash-attention/releases/tag/v2.5.5
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install pandas


cd ../libero-env/
pip install -e LIBERO-original/.
pip install -r LIBERO-original/libero_requirements.txt

pip install -e LIBERO-plus/.
pip install -r LIBERO-plus/libero_requirements.txt
pip install mujoco==3.3.2