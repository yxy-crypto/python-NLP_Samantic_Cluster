import pandas as pd
from nltk.corpus import wordnet as wn
from tqdm import tqdm  # 导入tqdm
#从word net取出来的同义词在不同词性下有明显变化,但是有时候不如word2vec更加语义完整
# 读取Excel中的词表
wordlist_df = pd.read_excel('wordlist.xlsx')

# 创建一个空的DataFrame来存储输出
output_data = []


# 函数：根据单词获取同义词、词性、释义和语域
def get_synonyms_and_details(word):
    # 获取该单词的所有同义词集
    synsets = wn.synsets(word)
    result = []

    for synset in synsets:
        pos = synset.pos()  # 词性
        definition = synset.definition()  # 释义
        lexname = synset.lexname()  # 语域

        # 获取同义词
        synonyms = [lemma.name() for lemma in synset.lemmas()]

        # 结果存储
        result.append({
            'Word': word,
            'POS': pos,
            'Definition': definition,
            'Synonyms': ', '.join(synonyms),
            'Lexname (Domain)': lexname
        })

    return result


# 使用 tqdm 包裹遍历词表的过程，添加进度条
for word in tqdm(wordlist_df['word'], desc="Processing words", unit="word"):
    details = get_synonyms_and_details(word)
    output_data.extend(details)

# 将结果转换为DataFrame
output_df = pd.DataFrame(output_data)

# 将输出写入Excel文件
output_df.to_excel('WordnetSynonym.xlsx', index=False)

print("WordNet同义词数据已成功导出到 WordnetSynonym.xlsx")
