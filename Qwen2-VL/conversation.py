import json
import os
from tqdm import tqdm
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# 路径设置
base_model_path = "/root/autodl-tmp/Qwen2-VL/models/Qwen2-VL-7B-Instruct"
fine_tuned_model_path = "/root/autodl-tmp/Qwen2-VL/output/Qwen2-VL-7B-Lora/checkpoint-9372"   #add your finetuned model path
input_json_path = ""   #add your input json file path
output_dir = "./playground/output"
image_root = "./playground"

# 多个 temperature 值
temperature_list = [0.2]

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 加载模型
model = Qwen2VLForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype="auto",
    device_map="auto"
)

#加载微调lora权重
model = PeftModel.from_pretrained(
    model,
    fine_tuned_model_path,
    is_trainable=False
)


model.eval()

# 加载 Processor
processor = AutoProcessor.from_pretrained(base_model_path)

# 加载输入数据
with open(input_json_path, "r") as f:
    data = json.load(f)

# 为每个 temperature 分别存储结果
temp_to_results = {temp: [] for temp in temperature_list}

# 推理主循环
for item in tqdm(data):
    try:
        image_path = os.path.join(image_root, item["image"])
        conversations = item["conversations"]
        if len(conversations) < 2:
            continue

        user_msg = conversations[0]["value"].replace("<image>\n", "").strip()
        ground_truth = conversations[1]["value"]

        # 构造消息
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_msg}
            ]
        }]

        # 构建输入
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        for temp in temperature_list:
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=temp,
                    num_beams=1,               # 设置为1，否则 Beam Search 会覆盖掉 temperature 效果
                    top_p=0.9,          # 可选：增加多样性
                    #length_penalty = 1.2
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]

            temp_to_results[temp].append({
                "id": item["id"],
                "question": user_msg,
                "ground_truth": ground_truth,
                "generated": output_text.strip(),
                "temperature": temp
            })

    except Exception as e:
        print(f"[ERROR] Failed to process id {item.get('id')}: {str(e)}")

# 分别写入不同文件
for temp, results in temp_to_results.items():
    file_path = os.path.join(output_dir, f"fine_tune_member_group39_T{temp}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ 所有温度的推理结果已保存到：{output_dir}")
