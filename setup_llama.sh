# Setup environment for prompting Llama3.1 in VinaBench data annotation
# This setup is based on what already built by setup_baseline.sh (vina_ft)

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

conda create -n vina_llama --clone vina_ft
conda activate vina_llama
pip install transformers==4.43.1 numpy==1.26.4
python -m nltk.downloader 'punkt'
