# DP-MIA: Dual-Phase Membership Inference Attack on Vision-Language Models across Pretraining and Finetuning Stages

This repository contains the experimental code for the paper **"DP-MIA: Dual-Phase Membership Inference Attack on Vision-Language Models across Pretraining and Finetuning Stages"**. Below are instructions for setting up the environment, training Vision-Language Models (VLMs), generating conversation outputs, and performing Membership Inference Attacks.

## VLM Training

This experiment focuses on two popular VLMs: [LLaVA](https://github.com/haotian-liu/LLaVA/tree/v1.0.1) and [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL). You will need to set up the environment, download pre-trained weights, split datasets, and train the models (instruction tuning stage only).

### LLaVA

To quickly setup the environment,you can follow the [LLaVA-Train](https://github.com/haotian-liu/LLaVA/tree/v1.0.1?tab=readme-ov-file#train) instructions to configure checkpoints and parameters.


### MiniGPT-4

To quickly setup the environment,you can follow the [Qwen2-VL-Train](https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-finetune/README.md) instructions to configure checkpoints and parameters.
