import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from paddlenlp import Taskflow

# seg = Taskflow("word_segmentation",home_path= "/paddle")
ner = Taskflow("ner",entity_only=True,home_path= "/paddle")
def my_cut(text):
    my_list = ner(text)
    for item in my_list:
        yield item[0]
# res = seg("编程(programming)，可简单理解为，程序员与计算机进行交流的过程。程序员使用计算机语言告诉计算机做什么，怎么做，计算机也将做事的过程和结果反馈给程序员，因此，先从程序员与计算机的交流场景开始，学习编程。")
# 定义文本列表
texts = []
indexs = ["第一章", "第二章", "第三章", "第四章", "第五章"]
for index in indexs:
    with open("./data/raw/" + index + ".txt", 'r', encoding='utf-8') as fr:
        lines = [line.strip() for line in fr]
        result_string = ''.join(lines)
        texts.append(result_string)
# 创建一个TfidfVectorizer对象，指定分词器为 jieba 分词器，并设置停用词表
vectorizer = TfidfVectorizer(tokenizer=my_cut)

# 对文本进行向量化
X = vectorizer.fit_transform(texts)

# 获取所有单词的列表
words_list = vectorizer.get_feature_names_out()

# print("words_list",words_list)

# 查找目标单词在单词列表中的索引
target_word_index = words_list.tolist().index('表达式') #第一章有表达式，第二章也有表达式

# print("target_word_index",target_word_index)
# 对TF-IDF矩阵进行归一化
X_normalized = normalize(X, norm='l2', axis=1)

# print(X_normalized)
# 获取目标单词在TF-IDF矩阵中的权重值
target_word_weight1 = X_normalized[0, target_word_index]
target_word_weight2 = X_normalized[1, target_word_index]
# 计算每个文档中每个单词的相对出现频率
print("第一章中'表达式'的权重",target_word_weight1)
print("第二章中'表达式'的权重",target_word_weight2)




#续上ner任务
def printNers(ners, fw, index):
    ners = list(set(ners))
    weight = []
    res = [t[0] for t in ners if t[1] in ["术语类", "术语类_术语类型", "术语类_符号指标类"]]
    if len(res) != 0:
        fw.write(str(res))  # 输出上一节的实体
        res_list = []
        for word in res:
            wlist = words_list.tolist()
            if word in wlist:
                target_word_index = words_list.tolist().index(word)
                target_word_weight = X_normalized[index, target_word_index]
                weight.append(target_word_weight)
                if target_word_weight > 0.01:
                    res_list.append(word)
            else:
                weight.append(0)
        fw.write('\n')
        fw.write(str(weight))
        fw.write('\nres_list：\n')
        fw.write(str(res_list))
    fw.write('\n')


with open("./data/res/" + "ResWithWeight.txt", 'w', encoding='utf-8') as fw:
    # for t in res:
    #     fw.write(str(t)+'\n')
    ners = []
    titles = ["第一章", "第二章", "第三章", "第四章", "第五章"]
    for index,title in enumerate(titles):
        fw.write(title + '\n')
        fw.write("===========================" + '\n')
        ners = ner(texts[index])
        printNers(ners, fw, index)
        ners.clear()
        fw.write('\n')