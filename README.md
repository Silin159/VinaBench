# VinaBench
The official repository of VinaBench: Benchmark for Faithful and Consistent Visual Narratives.

## Getting Started
VinaBench environments are developed based on Ubuntu 22.04, CUDA 12.1, Python 3.10 and Conda.

### Scripts for setting up environments:
```
# for training visual narrative baselines
bash setup_baseline.sh

# for Mantis-Idefics2
bash setip_mantis.sh

# for Llama-3.1-70B-Instruct
bash setip_llama.sh

# for Llama-OneVision-72B
bash setip_llava_onev.sh

# for MiniCPM-V-2.6
bash setup_minicpm.sh

# for training LLM narrative constraint generators
bash setup_torchtune.sh
```

## Preparing VinaBench Data
Please follow `data/README.md` to prepare the VinaBench data.

## VinaBench Baseline Training and Inference
We have tested three baseline models on VinaBench:
- [MM-Interleaved](https://arxiv.org/abs/2401.10208): please follow `MM-Interleaved/README.md`
- [AR-LDM](https://arxiv.org/abs/2211.10950): coming soon...
- [StoryGen](https://arxiv.org/abs/2306.00973): coming soon...

## VinaBench Evaluation
coming soon...
