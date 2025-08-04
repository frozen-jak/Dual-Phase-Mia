# DP-MIA: Dual-Phase Membership Inference Attack on Vision-Language Models across Pretraining and Finetuning Stages

This repository contains the experimental code for the paper **"DP-MIA: Dual-Phase Membership Inference Attack on Vision-Language Models across Pretraining and Finetuning Stages"**. Below are instructions for setting up the environment, training Vision-Language Models (VLMs), generating conversation outputs, and performing Membership Inference Attacks.

## VLM Training

This experiment focuses on two popular VLMs: [LLaVA](https://github.com/haotian-liu/LLaVA/tree/v1.0.1) and [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL). You will need to set up the environment, download pre-trained weights, split datasets, and train the models (instruction tuning stage only).

### LLaVA

To quickly setup the environment,you can follow the [LLaVA-Train](https://github.com/haotian-liu/LLaVA/tree/v1.0.1?tab=readme-ov-file#train) instructions to configure checkpoints and parameters.

Although this MIA targets the both **Pretraining** stage and **Finetuning** stage,for save resoure,we only train the finetuning stage.After preparing the Vicuna/LLaMA checkpoints, you can skip the [Pretrain](https://github.com/haotian-liu/LLaVA/tree/v1.0.1?tab=readme-ov-file#pretrain-feature-alignment) step and directly download the pre-trained projectors from [LLaVA MODEL_ZOO](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md#projector-weights) for Visual Instruction Tuning.

Download the instruction tuning data: [llava_instruct_158k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json) and COCO train2017 images [here](https://cocodataset.org/#download).

To distinguish between member and non-member data for MIA, split the `llava_instruct_158k.json` randomly.It should be noted that the file contains a large number of duplicate image data sets (although their prompts are different, we believe this will affect the judgment of member attributes). Please ensure the consistency of the images of members and non-members when filtering. In our experiment, 80% of the data is used for training the model, and 20% is kept as non-member data.

Next, follow the [Visual Instruction Tuning](https://github.com/haotian-liu/LLaVA/tree/v1.0.1?tab=readme-ov-file#visual-instruction-tuning) instructions to train the model using member data.


### Qwen2-VL

To quickly setup the environment,you can follow the [Qwen2-VL-Train](https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-finetune/README.md) instructions to configure checkpoints and parameters and follow the step to complete the finetuning work.

## Raw data acquisition

### LLaVA

Use the 
