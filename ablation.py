import json
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

# 加载 sentence embedding 模型
embedder = SentenceTransformer('all-mpnet-base-v2')

PRE_MEMBER_JSON = ""
FINE_MEMBER_JSON = ""
NON_MEMBER_JSON = ""
#add your json files

def safe_get(data, key):
    v = data.get(key, None)
    if v is None or (isinstance(v, str) and v.strip() == ''):
        return None
    return v

def process_file(json_path, label):
    features_all = []
    labels = []
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    for item in tqdm(data_list, desc=f"Processing {json_path}"):
        runtime = safe_get(item, 'runtime')
        confidence = safe_get(item, 'confidence_score')
        generated = safe_get(item, 'generated')
        ground_truth = safe_get(item, 'ground_truth')

        if None in [runtime, confidence, generated, ground_truth]:
            continue
        if generated.strip() == '' or ground_truth.strip() == '':
            continue

        gen_len = len(generated.split())
        if gen_len == 0:
            continue

        avg_time = runtime / gen_len
        if avg_time <= 0 or np.isnan(avg_time) or np.isnan(confidence):
            continue

        try:
            emb_gen = embedder.encode(generated, convert_to_tensor=True)
            emb_gt = embedder.encode(ground_truth, convert_to_tensor=True)
            cosine_sim = float(util.cos_sim(emb_gen, emb_gt).item())
        except Exception as e:
            print(f"Embedding error: {e}")
            continue

        features_all.append([cosine_sim, avg_time, confidence, gen_len])
        labels.append(label)

    return features_all, labels

def main():
    pre_features, pre_labels = process_file(PRE_MEMBER_JSON, 0)
    fine_features, fine_labels = process_file(FINE_MEMBER_JSON, 1)
    non_features, non_labels = process_file(NON_MEMBER_JSON, 2)

    X_all = np.array(pre_features + fine_features + non_features)
    y_all = np.array(pre_labels + fine_labels + non_labels)

    feature_combinations = {
        "CS + T": [0, 1],
        "CS + C": [0, 2],
        "CS + L": [0, 3],
        "T + C + L": [1, 2, 3],
        "CS + T + C": [0, 1, 2],
        "CS + T + L": [0, 1, 3],
        "CS + C + L": [0, 2, 3]
    }

    precisions, recalls, accuracies = [], [], []

    for name, indices in feature_combinations.items():
        X = X_all[:, indices]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_all, test_size=0.2, random_state=42, stratify=y_all)

        clf = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        acc = accuracy_score(y_test, y_pred)

        precisions.append(prec)
        recalls.append(rec)
        accuracies.append(acc)

        print(f"{name} - Precision: {prec:.4f}, Recall: {rec:.4f}, Accuracy: {acc:.4f}")

    # Sort the results by Precision (or any metric you want to sort by)
    sort_order = np.argsort(precisions)  # Sorting by Precision; use recalls/accuracies to sort by others
    sorted_precisions = np.array(precisions)[sort_order]
    sorted_recalls = np.array(recalls)[sort_order]
    sorted_accuracies = np.array(accuracies)[sort_order]
    sorted_labels = np.array(list(feature_combinations.keys()))[sort_order]

    y_pos = np.arange(len(sorted_labels))
    bar_width = 0.25

    plt.rcParams.update({
        'font.size': 14,  # 默认字体大小
        'axes.titlesize': 18,  # 标题
        'axes.labelsize': 18,  # 坐标轴标签
        'xtick.labelsize': 18,  # x刻度
        'ytick.labelsize': 18,  # y刻度
        'legend.fontsize': 18,  # 图例文字
        'legend.title_fontsize': 18  # 图例标题
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y_pos - bar_width, sorted_precisions, bar_width, label='Precision')
    ax.barh(y_pos, sorted_recalls, bar_width, label='Recall')
    ax.barh(y_pos + bar_width, sorted_accuracies, bar_width, label='Accuracy')

    # Add data labels on top of the bars
    for i, v in enumerate(sorted_precisions):
        ax.text(v + 0.01, i - bar_width, f'{v:.2f}', color='black', va='center')
    for i, v in enumerate(sorted_recalls):
        ax.text(v + 0.01, i, f'{v:.2f}', color='black', va='center')
    for i, v in enumerate(sorted_accuracies):
        ax.text(v + 0.01, i + bar_width, f'{v:.2f}', color='black', va='center')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels)
    ax.set_xlabel("Score")
    ax.set_title("Ablation Study on RIGEL + Cosine Similarity Features on  LLaVA-v1.5-7b")
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    plt.tight_layout()
    output_image_path = 'llava-7b-data/ablation experiment/ablation_study_result.png'  # 可以修改为你想要保存的路径和格式
    plt.savefig(output_image_path, bbox_inches='tight')  # bbox_inches='tight' 用于去掉多余的空白边距
    print(f"图像已保存为 {output_image_path}")
    plt.show()

if __name__ == "__main__":
    main()
