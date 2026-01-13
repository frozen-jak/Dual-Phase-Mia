import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel


############################################
# 1. 配置区（只需要改这里）
############################################

# JSON 输入路径
#数据采用LLAVA-7B的
JSON_PATH = r"D:\MIAexp\Dual-Phase-MIA\Qwen2b-vl\generate_data_1\nonmember.json"

# 输出 JSON 路径
SAVE_PATH = r"D:\MIAexp\Dual-Phase-MIA\LOMIA\qwen-2b\lomia_nonmember.json"

# 图片文件夹路径（图片名 = id + .jpg / .png）
IMAGE_DIR = r"D:\MIAexp\Test\train2017" #微调成员、非成员图片文件夹

#IMAGE_DIR = r"D:\MIAexp\Test\llava_500_images\images_pretrained_member10k"

# 图片后缀（根据你的数据改）
IMAGE_SUFFIX = ".jpg"

# 使用设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


############################################
# 2. 加载模型（只加载一次）
############################################

print("Loading T2T model (Sentence-BERT)...")
t2t_model = SentenceTransformer(
    "all-mpnet-base-v2",
    device=DEVICE
)

print("Loading CLIP model for I2T...")
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14"
).to(DEVICE)

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-large-patch14"
)

clip_model.eval()


############################################
# 3. 相似度计算函数
############################################

def compute_t2t_similarity(text_a: str, text_b: str) -> float:
    """
    Text-to-Text cosine similarity
    """
    embeddings = t2t_model.encode(
        [text_a, text_b],
        normalize_embeddings=True
    )
    return float(np.dot(embeddings[0], embeddings[1]))


def compute_i2t_similarity(image_path: str, text: str) -> float:
    """
    Image-to-Text cosine similarity using CLIP
    """
    image = Image.open(image_path).convert("RGB")

    inputs = clip_processor(
        images=image,
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    ).to(DEVICE)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_emb = outputs.image_embeds
        text_emb = outputs.text_embeds

    # L2 normalize
    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    similarity = image_emb @ text_emb.T
    return float(similarity.item())


############################################
# 4. 主处理流程
############################################

def main():
    print(f"Loading JSON from: {JSON_PATH}")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")
    print("Computing T2T and I2T similarities...")

    for item in tqdm(data):
        # -------- 文本 --------
        ground_truth = item["ground_truth"]
        generated = item["generated"]

        # -------- 图片路径 --------
        image_id = item["id"]
        image_path = os.path.join(
            IMAGE_DIR,
            image_id + IMAGE_SUFFIX
        )

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # -------- T2T --------
        item["T2T"] = compute_t2t_similarity(
            ground_truth,
            generated
        )

        # -------- I2T --------
        item["I2T"] = compute_i2t_similarity(
            image_path,
            generated
        )

    print(f"Saving results to: {SAVE_PATH}")
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Done ✔")


############################################
# 5. 程序入口
############################################

if __name__ == "__main__":
    main()
