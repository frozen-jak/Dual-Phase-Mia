import json
import re

def extract_confidence(text):
    """
    从最后一句中提取 confidence score，并删除该句
    """
    if not text or not text.strip():
        return None, text

    text = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)  # 按句子分割（保留 .!? 后的边界）
    if not sentences:
        return None, text

    last_sentence = sentences[-1]

    patterns = [
        r'(?:my\s+)?confidence\s+score\s+(?:is|of)?\s*[:\-]?\s*(\d{1,3})[\.\s]*$',   # My confidence score is 95.
        r'(\d{1,3})\s*out of\s*100[\.\s]*$',                                          # 95 out of 100
        r'confidence\s*[:\-]?\s*(\d{1,3})[\.\s]*$',                                   # confidence: 95
        r'\bconfidence.*?(\d{1,3})[\.\s]*$',                                          # catch-all
    ]

    for pattern in patterns:
        match = re.search(pattern, last_sentence, re.IGNORECASE)
        if match:
            try:
                score = int(match.group(1))
                if 0 <= score <= 100:
                    # 删除最后一句
                    cleaned_text = ' '.join(sentences[:-1]).strip()
                    return score, cleaned_text
            except:
                continue

    return None, text


def process_json_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i, item in enumerate(data):
        gen_text = item.get("generated", "")

        score, cleaned = extract_confidence(gen_text)

        if score is not None:
            item["confidence_score"] = score
            item["generated"] = cleaned
        else:
            item["confidence_score"] = None
            item["generated"] = gen_text

        # 日志：检测生成为空的情况
        if not item["generated"].strip():
            print(f"[警告] 第{i}项生成内容为空，ID: {item.get('id')}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 使用方法

# 示例使用
input_file = 'llava-7b-data/ablation experiment/generate_data_1/nonmember_group3_0.2.json'
output_file = 'llava-7b-data/ablation experiment/data_confidence_time/nonmember_group3.json'
process_json_file(input_file, output_file)
