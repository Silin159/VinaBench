# Setup environment for VinaBench baseline model finetuning

# Development platform:
#   linux/amd64 nvidia/cuda:12.1.0-devel-ubuntu22.04

# Basic requirements:
#   build-essential
#   cmake
#   g++
#   git
#   curl
#   vim
#   unzip
#   wget
#   tmux
#   screen
#   ca-certificates
#   apt-utils
#   libjpeg-dev
#   libpng-dev
#   python3.10
#   python3.10-dev
#   python3.10-distutils
#   python3-pip
#   python3-setuptools
#   librdmacm1
#   libibverbs1
#   ibverbs-providers
#   ffmpeg
#   libsm6
#   libxext6

# Conda installation:
# bash ./miniconda.sh

conda create -n vina_ft python=3.10
conda activate vina_ft
pip install --upgrade pip setuptools wheel
pip install -r requirements_baseline.txt
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install flash-attn --no-build-isolation
