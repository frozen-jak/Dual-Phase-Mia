import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize


# ===============================
# 1. IO & Feature utils
# ===============================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_proxy_features(data, label):
    X, y_ppl, y_cls = [], [], []
    for item in data:
        X.append([item["I2T"], item["T2T"]])
        y_ppl.append(item["ppl"])
        y_cls.append(label)
    return np.array(X), np.array(y_ppl), np.array(y_cls)


def extract_target_features(data, label):
    X, y_cls = [], []
    for item in data:
        X.append([item["I2T"], item["T2T"]])
        y_cls.append(label)
    return np.array(X), np.array(y_cls)


# ===============================
# 2. Main
# ===============================
def main():
    # -------- Proxy 数据路径 --------
    proxy_member_path = r"D:\MIAexp\Dual-Phase-MIA\LOMIA\LOMIA_PROXY_RESULT\lomia_finetune_member.json"
    proxy_pretrain_path = r"D:\MIAexp\Dual-Phase-MIA\LOMIA\LOMIA_PROXY_RESULT\lomia_pretrained_member.json"
    proxy_nonmember_path = r"D:\MIAexp\Dual-Phase-MIA\LOMIA\LOMIA_PROXY_RESULT\lomia_nonmember.json"

    # -------- Target 数据路径 --------
    target_member_path = r"D:\MIAexp\Dual-Phase-MIA\real_world_attack\privacy protection\LOMIA\finetuned_member.json"
    target_pretrain_path = r"D:\MIAexp\Dual-Phase-MIA\real_world_attack\privacy protection\LOMIA\pretrained_member.json"
    target_nonmember_path = r"D:\MIAexp\Dual-Phase-MIA\real_world_attack\privacy protection\LOMIA\nonmember.json"


    print("🔹 Loading proxy datasets...")
    proxy_member = load_json(proxy_member_path)
    proxy_pretrain = load_json(proxy_pretrain_path)
    proxy_nonmember = load_json(proxy_nonmember_path)

    # ========== Regression (unchanged) ==========
    X_m, y_m_ppl, y_m_cls = extract_proxy_features(proxy_member, 2)
    X_p, y_p_ppl, y_p_cls = extract_proxy_features(proxy_pretrain, 1)
    X_n, y_n_ppl, y_n_cls = extract_proxy_features(proxy_nonmember, 0)

    X_proxy = np.vstack([X_m, X_p, X_n])
    y_proxy_ppl = np.concatenate([y_m_ppl, y_p_ppl, y_n_ppl])
    y_proxy_cls = np.concatenate([y_m_cls, y_p_cls, y_n_cls])

    print("🔹 Fitting LOMIA regressor...")
    reg = LinearRegression()
    reg.fit(X_proxy, y_proxy_ppl)

    A, B = reg.coef_
    C = reg.intercept_

    print(f"✅ Regressor learned: A={A:.6f}, B={B:.6f}, C={C:.6f}")

    # ========== Proxy-based threshold learning ==========
    print("🔹 Searching thresholds on proxy data...")

    ppl_proxy_pred = reg.predict(X_proxy)

    candidates = np.percentile(ppl_proxy_pred, np.linspace(5, 95, 50))
    best_f1 = -1
    best_t1, best_t2 = None, None

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            t1, t2 = candidates[i], candidates[j]
            y_pred = []

            for p in ppl_proxy_pred:
                if p <= t1:
                    y_pred.append(2)
                elif p <= t2:
                    y_pred.append(1)
                else:
                    y_pred.append(0)

            f1 = f1_score(y_proxy_cls, y_pred, average="macro")

            if f1 > best_f1:
                best_f1 = f1
                best_t1, best_t2 = t1, t2

    print(f"✅ Best proxy thresholds:")
    print(f"   τ1 = {best_t1:.6f}, τ2 = {best_t2:.6f}")
    print(f"   Proxy macro-F1 = {best_f1:.4f}")

    # ========== Target inference ==========
    print("🔹 Loading target datasets...")
    target_member = load_json(target_member_path)
    target_pretrain = load_json(target_pretrain_path)
    target_nonmember = load_json(target_nonmember_path)

    X_tm, y_tm = extract_target_features(target_member, 2)
    X_tp, y_tp = extract_target_features(target_pretrain, 1)
    X_tn, y_tn = extract_target_features(target_nonmember, 0)

    X_target = np.vstack([X_tm, X_tp, X_tn])
    y_true = np.concatenate([y_tm, y_tp, y_tn])

    ppl_target_pred = reg.predict(X_target)

    y_pred = []
    for p in ppl_target_pred:
        if p <= best_t1:
            y_pred.append(2)
        elif p <= best_t2:
            y_pred.append(1)
        else:
            y_pred.append(0)

    y_pred = np.array(y_pred)

    # ========== Metrics ==========
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    accuracy = accuracy_score(y_true, y_pred)

    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    score = -ppl_target_pred

    auc = roc_auc_score(
        y_true_bin,
        np.vstack([score, score, score]).T,
        average="macro",
        multi_class="ovr"
    )

    print("\n📊 LOMIA (Proxy-threshold Transfer) Results:")
    print(f"   Precision (macro): {precision:.4f}")
    print(f"   Recall    (macro): {recall:.4f}")
    print(f"   Accuracy         : {accuracy:.4f}")
    print(f"   AUC       (macro): {auc:.4f}")


if __name__ == "__main__":
    main()
