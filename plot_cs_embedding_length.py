import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer  # 新增

print('loading model...')
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  # 更高精度
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')  # 新增tokenizer加载

def semantic_similarity(text1, text2):
    print('calculating semantic similarity......')
    embeddings = model.encode([text1, text2])
    return util.cos_sim(embeddings[0], embeddings[1]).item()

# 1. 数据加载和预处理
def load_data(file_path, label):
    with open(file_path, 'r',encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['group'] = label
    return df

# 2. 特征计算（相似度和token长度）
def calculate_features(row):
    print('calculating features...')
    # 用token数量替代词数
    gt_len = len(tokenizer.tokenize(row['ground_truth']))
    gen_len = len(tokenizer.tokenize(row['generated']))
    cs = semantic_similarity(row['ground_truth'], row['generated'])
    return pd.Series({
        'length_diff': gen_len - gt_len,
        'cosine_sim': cs,
        'gt_length': gt_len,
        'gen_length': gen_len
    })

if __name__ == '__main__':
    print('Loading data...')
    df_pretrain = load_data('llava-7b-llama-chat-data/generate_data1/pretrained_member_group1_0.2.json', 'pretrain_member')
    df_nonmember = load_data('llava-7b-llama-chat-data/generate_data1/nonmember_group3_0.2.json', 'nonmember')
    df_finetune = load_data('llava-7b-llama-chat-data/generate_data1/finetune_member_group39_0.2.json', 'finetune_member')

    print('merging data...')
    df = pd.concat([df_pretrain, df_nonmember, df_finetune], ignore_index=True)

    df = df.join(df.apply(calculate_features, axis=1))

    # 3. 统计分析
    stats = df.groupby(['group', 'temperature']).agg({
        'cosine_sim': ['mean', 'std', 'count'],
        'length_diff': ['mean', 'std'],
        'gen_length': ['mean', 'std']
    }).round(3)

    print("统计摘要:")
    print(stats)

    # 4. 可视化分析
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x='group', y='cosine_sim', hue='temperature')
    plt.title('Cosine Similarity Distribution')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 2)
    sns.violinplot(data=df, x='group', y='length_diff', hue='temperature', split=True)
    plt.title('Generation Length Difference (Gen - GT)')

    plt.subplot(2, 2, 4)
    length_pivot = df.pivot_table(index='group', columns='temperature',
                                  values='gen_length', aggfunc='mean')
    sns.heatmap(length_pivot, annot=True, fmt=".1f", cmap='YlOrRd')
    plt.title('Average Generation Embedding Length')

    plt.tight_layout()
    plt.savefig('llava-7b-llama-chat-data/results_1/member_analysis_cs_embedding_length_group1_T0.2.png', dpi=300)
    plt.show()

    # 5. 保存结果
    df.to_csv('llava-7b-llama-chat-data/results_1/all_data_with_features_cs_embedding_length_group1_T0.2.csv', index=False)

    print("分析结果已保存到:")
    print("- member_analysis.png (可视化图表)")
    print("- all_data_with_features.csv (带特征的全部数据)")
