# -*- coding: utf-8 -*-
"""
改进版语义聚类分析脚本 (Python 3.7兼容版)
修复内容：
1. 显式构建词汇表
2. 优化噪声点处理
3. 增强异常处理
"""

# 安装依赖（终端执行）：
# pip install gensim==3.8.3 umap-learn==0.5.3 hdbscan==0.8.28 scikit-learn==0.24.2 pandas==1.3.5 nltk==3.6.3

import re
import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from umap import UMAP
from hdbscan import HDBSCAN, approximate_predict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from collections import defaultdict
import logging

nltk.download('stopwords')
nltk.download('wordnet')

# 配置日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 配置参数
INPUT_PATH = "ArticleCluster.txt"  # 输入文件路径
OUTPUT_CSV = "clusters_semantic.csv"  # 聚类结果文件
KEYWORDS_CSV = "cluster_keywords.csv"  # 关键词文件
OUTPUT_DIR = "clusters_word"  # 簇文本输出目录
WINDOW_SIZE = 10  # 词向量窗口大小
MIN_COUNT = 5  # 最小词频阈值
CLUSTER_SIZE = 10  # 最小簇大小
CLUSTER_EPS = 0.6  # 簇合并敏感度
VECTOR_SIZE = 150  # 词向量维度
TRAIN_EPOCHS = 10  # 训练迭代次数


def clean_text(text):
    """改进的文本清洗函数"""
    text = re.sub(r'[^a-zA-Z\s-]', '', text)
    words = text.lower().split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    cleaned = []
    for w in words:
        if len(w) <= 2 or w in stop_words:
            continue
        # 处理连字符词汇
        if '-' in w:
            parts = [lemmatizer.lemmatize(p) for p in w.split('-') if len(p) > 2]
            cleaned.extend(parts)
        else:
            cleaned.append(lemmatizer.lemmatize(w))
    return cleaned


def generate_keywords(cluster_texts, freq_dict):
    """生成加权关键词(TF-IDF 60% + 词频40%)"""
    if not cluster_texts:
        return []

    try:
        tfidf = TfidfVectorizer(max_df=0.8).fit_transform(cluster_texts)
        features = tfidf.get_feature_names_out()
    except ValueError:
        return ['insufficient_data'] * len(cluster_texts)

    keywords = []
    for i in range(tfidf.shape[0]):
        row = tfidf[i].toarray().flatten()
        freq_scores = np.array([freq_dict.get(f, 0) for f in features])
        if freq_scores.max() > 0:
            freq_scores = freq_scores / freq_scores.max()
        else:
            freq_scores = np.zeros_like(freq_scores)
        combined = 0.6 * row + 0.4 * freq_scores
        top_idx = combined.argsort()[-5:][::-1]
        keywords.append(', '.join(features[top_idx]))
    return keywords


def export_clusters(df, keywords_df):
    """将每个簇的词汇导出到独立文件"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    name_map = {}
    for _, row in keywords_df.iterrows():
        keys = [k.strip() for k in row['keywords'].split(',')[:3]]
        clean_name = '_'.join(keys).replace(' ', '')
        name_map[row['cluster']] = re.sub(r'[^\w-]', '', clean_name)[:40]

    for cluster_id in df['cluster'].unique():
        if cluster_id == -1:
            continue

        cluster_words = df[df['cluster'] == cluster_id]
        if cluster_words.empty:
            continue

        sorted_words = cluster_words.sort_values(
            'frequency', ascending=False)['word'].tolist()

        base_name = name_map.get(cluster_id, f"cluster_{cluster_id}")
        filename = os.path.join(
            OUTPUT_DIR,
            f"cluster_{cluster_id}_{base_name}.txt"
        )

        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted_words))


def main():
    # 1. 数据清洗和词频统计
    word_freq = defaultdict(int)
    corpus = []
    try:
        with open(INPUT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                cleaned = clean_text(line.strip())
                if not cleaned:  # 跳过空行处理结果
                    continue
                for word in cleaned:
                    word_freq[word] += 1
                corpus.append(cleaned)
    except FileNotFoundError:
        print(f"错误：输入文件 {INPUT_PATH} 不存在")
        return

    if not corpus:
        print("错误：清洗后的语料为空，请检查输入文件和清洗逻辑")
        return

    # 2. 训练词向量模型
    model = Word2Vec(
        vector_size=VECTOR_SIZE,
        window=WINDOW_SIZE,
        min_count=MIN_COUNT,
        workers=4,
        hs=1,
        sample=1e-5
    )

    # 显式构建词汇表
    model.build_vocab(corpus)
    if not model.wv.key_to_index:
        print("错误：词汇表为空，请降低min_count阈值或检查输入数据")
        return

    # 显式训练模型
    model.train(
        corpus,
        total_examples=model.corpus_count,
        epochs=TRAIN_EPOCHS,
        compute_loss=True
    )

    vocab = list(model.wv.key_to_index.keys())
    vectors = model.wv.vectors

    # 3. 降维处理
    reducer = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.2,
        metric='cosine',
        low_memory=True,
        random_state=42
    )
    vectors_2d = reducer.fit_transform(vectors)

    # 4. 密度聚类
    clusterer = HDBSCAN(
        min_cluster_size=CLUSTER_SIZE,
        cluster_selection_epsilon=CLUSTER_EPS,
        min_samples=2,
        prediction_data=True
    )
    clusters = clusterer.fit_predict(vectors)

    # 5. 处理噪声点
    if -1 in clusters:
        try:
            soft_labels, _ = approximate_predict(clusterer, vectors)
            clusters = np.where(clusters == -1, soft_labels, clusters)
        except Exception as e:
            print(f"噪声点处理失败：{str(e)}")

    # 6. 构建结果数据集
    result_df = pd.DataFrame({
        'word': vocab,
        'x': vectors_2d[:, 0],
        'y': vectors_2d[:, 1],
        'cluster': clusters,
        'frequency': [word_freq[w] for w in vocab]
    })

    # 7. 提取关键词
    valid_clusters = [c for c in np.unique(clusters) if c != -1]
    cluster_texts = []
    for c in valid_clusters:
        words = result_df[result_df['cluster'] == c]['word'].tolist()
        cluster_texts.append(' '.join(words))

    keywords_df = pd.DataFrame()
    if cluster_texts:
        try:
            keywords = generate_keywords(cluster_texts, word_freq)
            keywords_df = pd.DataFrame({
                'cluster': valid_clusters,
                'keywords': keywords,
                'word_count': [len(ct.split()) for ct in cluster_texts]
            })
            keywords_df.to_csv(KEYWORDS_CSV, index=False)
        except Exception as e:
            print(f"关键词生成失败：{str(e)}")

    # 8. 导出结果
    try:
        result_df.to_csv(OUTPUT_CSV, index=False)
        if not keywords_df.empty:
            export_clusters(result_df, keywords_df)
            print(f"成功导出{len(valid_clusters)}个簇到目录: {OUTPUT_DIR}")
        else:
            print("警告：未生成有效聚类")
    except PermissionError:
        print("错误：文件写入权限被拒绝，请检查输出路径")


if __name__ == "__main__":
    main()