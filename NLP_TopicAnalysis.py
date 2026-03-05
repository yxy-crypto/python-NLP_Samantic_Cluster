from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import os

# 强制离线，不让它联网尝试下载模型
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["NO_PROXY"] = "huggingface.co"

# 加载本地模型（必须是下载好的目录）
model_path = "C:/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(model_path)

# 初始化 BERTopic，显式使用本地 embedding_model
topic_model = BERTopic(embedding_model=embedding_model, language="english", calculate_probabilities=True)

# 示例文本（可替换为多个文档）
docs = [
    "Apple releases new iPhone with exciting features.",
    "The stock market showed signs of recovery today.",
    "Ukraine faces ongoing challenges due to the war.",
    "Machine learning continues to revolutionize tech.",
    "Apple launches new product.",
    "Ukraine conflict continues.",
    "Stock markets are volatile.",
    "Climate change needs action.",
    "AI is transforming society.",
    "COVID-19 changed the world.",
    "Cryptocurrency is volatile.",
    "Education is evolving.",
    "SpaceX launches again.",
    "Mental health awareness grows.",
    "Water scarcity is rising.",
    "Electric vehicles expand.",
    "Privacy concerns grow online.",
    "Social media addiction.",
    "Remote work becomes norm.",
    "Inflation hits hard.",
    "Healthcare systems stressed.",
    "Tech layoffs in 2024.",
    "War impacts economy.",
    "Elections influence markets.",
    "Environmental issues demand urgent global action."

]


# 拟合模型
topics, probs = topic_model.fit_transform(docs)

# 输出主题信息
for i, topic in enumerate(topics):
    print(f"Document {i}: Topic {topic}")
    print(topic_model.get_topic(topic))

# 可视化主题频率
topic_model.visualize_barchart(top_n_topics=5).show()



