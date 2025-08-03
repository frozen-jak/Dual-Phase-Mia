import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# 读取数据
df = pd.read_csv("Qwen7b-vl/results_1/all_data_with_features_cs_embedding_length_group1_T0.2.csv")

print(df["group"])
# 替换 group 字段为易读格式
df["group"] = df["group"].replace({
    "pretrain_member": "Pre-member",
    "finetune_member": "Fine-member",
    "nonmember": "Non-member"
})

# 设置绘图风格
sns.set(style="whitegrid")

# 初始化图形

plt.rcParams.update({
    'font.size': 18,            # 默认字体大小
    'axes.titlesize': 18,       # 标题
    'axes.labelsize': 18,       # 坐标轴标签
    'xtick.labelsize': 18,      # x刻度
    'ytick.labelsize': 18,      # y刻度
    'legend.fontsize': 18,      # 图例文字
    'legend.title_fontsize': 18 # 图例标题
})


plt.figure(figsize=(10, 6))

# 修正后的 group 映射
group_colors = {
    "Pre-member": "C0",
    "Fine-member": "C1",
    "Non-member": "C2"
}

# 绘图，每类 group 单独绘制
for group, color in group_colors.items():
    subset = df[df["group"] == group]
    if subset.empty:
        print(f"[警告] 组 {group} 没有数据，跳过绘制")
        continue
    sns.histplot(
        subset["cosine_sim"],
        bins=200,
        stat="density",
        element="step",
        label=group,
        color=color,
        kde=True
    )

# 设置横坐标范围
plt.xlim(left=0.35,right=0.99)
plt.ylim(0,12)

plt.gca().xaxis.set_major_locator(MultipleLocator(0.05))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# 设置标题和坐标轴
plt.title("Embedding Cosine similarity histogram on Qwen2-vl-7b")
plt.xlabel("Cosine Similarity")
plt.ylabel("Density")

# 添加图例
plt.legend(title="Group", loc="upper left")

# 自动调整布局
plt.tight_layout()

#保存图片
plt.savefig("Qwen7b-vl/results_1/Cosine similarity histogram.pdf", bbox_inches='tight')

# 显示图像
plt.show()
