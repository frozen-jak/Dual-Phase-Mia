import json
import os
import torch
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.eval.eval_infer import eval_model_single_image
from tqdm import tqdm  # ✅ 用于显示进度条
import time
T = 0.1
#TOP_P = 0.9  # ✅ 设置 top_p 限制尾部采样
image_file_base = "./playground/data/images_pretrained_member"
input_dir = "./playground/data/input/pretrained_member_group.json"
output_dir = "./playground/data/out_put_new"
os.makedirs(output_dir, exist_ok=True)
BEAMS_LIST = [2, 4, 8]
TEMPERATURE_LIST = [0.2]
def clean_image_tag(text):
    return text.replace("<image>\n", "").replace("<image>", "")

def clean_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def extract_qa_pairs(conversations):
    qa_pairs = []
    for i in range(0, len(conversations) - 1, 2):
        if conversations[i]["from"] == "human" and conversations[i + 1]["from"] == "gpt":
            question = clean_image_tag(conversations[i]["value"])
            qa_pairs.append({
                "question": question,
                "answer": conversations[i + 1]["value"]
            })
    return qa_pairs

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def main():
    print("🔄 Loading model into memory...")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path="./model/llava-v1.5-7b-lora",
        model_base="./model/vicuna-7b-v1.5",
        model_name="llava-v1.5-7b-lora",
    )

    data = load_data(input_dir)

    for T in TEMPERATURE_LIST:
        print(f"\n🚀 Starting evaluation with temperature = {T}")
        results = []
        total = len(data)

        for idx, item in enumerate(tqdm(data, desc=f"[Temperature={T}] Processing images", ncols=100)):
            image_path = os.path.join(image_file_base, item["image"])
            image = Image.open(image_path).convert("RGB")
            temp = clean_image_tag(item["conversations"][0]["value"])
            prompt = f"{temp}Beside, please provide your confidence score out of 100 at the end of your response, formatted as a number. Do not include any extra words, just the score."
            print(prompt)
            print(f"🔹 Image {idx+1}/{total} | Temperature={T}")
            time1 = time.time()
            print(time1)
            try:
                output = eval_model_single_image(
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    image=image,
                    query=prompt,
                    conv_mode=None,
                    temperature=T,
                    top_p=None,
                    num_beams=None,
                    max_new_tokens=512
                )
            except RuntimeError as e:
                if "inf" in str(e).lower() or "nan" in str(e).lower():
                    print(f"⚠️ Skipping sample due to NaN/Inf error (Temperature={T})")
                    output = "[SKIPPED]"
                else:
                    raise e
            time2 = time.time()
            print(time2)
            runtime = time2 - time1
            results.append({
                    "id": item["id"],
                    "temperature": T,
                    #"num_beams": num_beams,
                    "question": prompt,
                    "ground_truth": clean_image_tag(item["conversations"][1]["value"]),
                    "generated": output,
                    "runtime": runtime
                })     

            if idx % 10 == 0:
                clean_memory()

        output_json = os.path.join(output_dir, f"pretrained_member_group1_{T}.json")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results for temperature={T} saved to: {output_json}")

    return

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
    torch.cuda.empty_cache()
