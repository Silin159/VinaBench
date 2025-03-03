# Setup environment for prompting MiniCPM in VinaBench data annotation and evaluation

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

conda create -n vina_minicpm python=3.10
conda activate vina_minicpm
pip install --upgrade pip setuptools wheel
pip install -r requirements_minicpm.txt
pip install flash-attn --no-build-isolation
