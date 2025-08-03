import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    f1_score
)
from scipy.stats import entropy
import seaborn as sns


def find_best_cosine_threshold_for_pretrain(df, verbose=True):
    best_f1 = 0
    best_t = 0.0
    label_bin = (df["label"] == 1).astype(int)

    for t in np.arange(0.3, 0.9, 0.01):
        pred = (df["cosine_sim"] < t).astype(int)
        f1 = f1_score(label_bin, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    if verbose:
        print(f"\n🔍 自动搜索最佳 cosine_sim 阈值: {best_t:.2f}（F1-score: {best_f1:.4f}）")
    return best_t

def run_two_step_classification(df, cosine_threshold, verbose=True):
    """
    两阶段分类器：
    STEP 1：使用 cosine 相似度识别预训练成员（label=1）
    STEP 2：在其余样本中，使用生成长度训练分类器区分非成员（0）和微调成员（2）
    """
    # 第一步：使用 cosine 相似度阈值判断是否为预训练成员
    df["pred_step1"] = np.where(df["cosine_sim"] < cosine_threshold, 1, -1)

    # 第二步：对剩余样本进行二分类（0 vs 2），基于 length_gen 训练
    mask_remaining = df["pred_step1"] == -1
    df_step2 = df[mask_remaining & df["label"].isin([0, 2])]

    X2 = df_step2[["gen_length"]]
    y2 = df_step2["label"]

    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, stratify=y2, test_size=0.2, random_state=42
    )

    clf2 = LogisticRegression()
    clf2.fit(X2_train, y2_train)
    y2_pred = clf2.predict(X2_test)

    # 回填第二阶段分类结果
    df.loc[df_step2.index, "pred_step2"] = clf2.predict(X2)
    df["final_pred_two_step"] = df["pred_step1"]
    df.loc[df["final_pred_two_step"] == -1, "final_pred_two_step"] = df["pred_step2"]

    # 评估整体三分类性能
    mask_valid = df["final_pred_two_step"].notna()
    y_true = df.loc[mask_valid, "label"]
    y_pred = df.loc[mask_valid, "final_pred_two_step"].astype(int)

    if verbose:
        print("\n📌 两阶段分类器结果：")
        print(classification_report(y_true, y_pred, target_names=["Non-member", "Pretrain", "Finetune"]))

    # 返回指标
    p, r, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return {"precision": p, "recall": r, "accuracy": acc}


def run_multiclass_classification(df, verbose=True):
    X = df[["gen_length", "cosine_sim"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.4, random_state=42
    )
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    if verbose:
        print("\n📌 多分类器评估结果:")
        print(classification_report(y_test, y_pred, target_names=["Non-member", "Pretrain", "Finetune"]))

    p, r, _, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    return {"precision": p, "recall": r, "accuracy": acc}



def group_average_features(df, group_size):
    grouped = []

    for label in sorted(df["label"].unique()):
        df_class = df[df["label"] == label].sample(frac=1, random_state=42).reset_index(drop=True)
        num_groups = len(df_class) // group_size

        for i in range(num_groups):
            group = df_class.iloc[i * group_size:(i + 1) * group_size]
            grouped.append({
                "gen_length": group["gen_length"].mean(),
                "cosine_sim": group["cosine_sim"].mean(),
                "label": label
            })

    return pd.DataFrame(grouped)


def plot_metrics(metrics_dict, title, output_name):
    group_sizes = list(metrics_dict.keys())
    precisions = [metrics_dict[g]["precision"] for g in group_sizes]
    recalls = [metrics_dict[g]["recall"] for g in group_sizes]
    accuracies = [metrics_dict[g]["accuracy"] for g in group_sizes]

    x = np.arange(len(group_sizes))
    width = 0.25

    plt.rcParams.update({
        'font.size': 18,  # 默认字体大小
        'axes.titlesize': 18,  # 标题
        'axes.labelsize': 18,  # 坐标轴标签
        'xtick.labelsize': 18,  # x刻度
        'ytick.labelsize': 18,  # y刻度
        'legend.fontsize': 18,  # 图例文字
        'legend.title_fontsize': 18  # 图例标题
    })

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width, precisions, width, label='Precision')
    bars2 = plt.bar(x, recalls, width, label='Recall')
    bars3 = plt.bar(x + width, accuracies, width, label='Accuracy')

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=13)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    plt.ylim(0.5, 0.92)
    plt.xticks(x, group_sizes)
    plt.xlabel("Group Size")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_name)
    plt.close()
    print(f"📊 已保存图像: {output_name}")


def main():
    df_raw = pd.read_csv("Qwen7b-vl/results_1/all_data_with_features_cs_embedding_length_group1_T0.2.csv")
    df_raw["label"] = df_raw["group"].map({
        "nonmember": 0,
        "pretrain_member": 1,
        "finetune_member": 2
    })

    group_sizes = [1, 2, 4, 8, 16]
    metrics_two_step = {}
    metrics_multiclass = {}

    for g in group_sizes:
        print(f"\n🧪 正在处理 group_size = {g}")
        df_grouped = group_average_features(df_raw, group_size=g)
        best_consine =find_best_cosine_threshold_for_pretrain(df_grouped, verbose=False)
        metrics_two_step[g] = run_two_step_classification(df_grouped.copy(),  cosine_threshold= best_consine, verbose=False)
        metrics_multiclass[g] = run_multiclass_classification(df_grouped.copy(), verbose=False)

    #pd.DataFrame(metrics_two_step).T.to_csv("metrics_two_step_group_1_2.csv")
    #pd.DataFrame(metrics_multiclass).T.to_csv("metrics_multiclass_group_1_2.csv")
    #print("\n📆 评估结果已保存成 CSV。")

    plot_metrics(metrics_two_step, " Dual-Binary Membership Inference Attack on Qwen2-VL-7B", "Qwen7b-vl/results_1/two_step_metrics_1.png")
    plot_metrics(metrics_multiclass, "Multi-class Membership Inference Attack on Qwen2-VL-7B", "Qwen7b-vl/results_1/multiclass_metrics_1.png")


if __name__ == "__main__":
    main()
