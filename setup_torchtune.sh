# Setup environment for finetuning LLM narrative constraint generator (for w/ LLM Cons. setting)

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

conda create -n vina_llm_cons python=3.10
conda activate vina_llm_cons
pip install --upgrade pip setuptools wheel
pip install torch==2.6.0 torchvision==0.21.0 torchao==0.8.0 torchtune==0.5.0
pip install transformers==4.43.1 accelerate tqdm
