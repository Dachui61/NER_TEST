import stanfordnlp

# 下载stanfordnlp的中文模型
# stanfordnlp.download('zh')

# 初始化stanfordnlp的中文分词器
nlp = stanfordnlp.Pipeline(lang='zh', processors='tokenize,mwt,pos,lemma,depparse')

# 导入jieba并设置词性标注模式为pseg
import jieba.posseg as pseg

def extract_relations(text):
    # 分别存储三种关系的结果
    subj_pred_obj_relations = []
    adj_post_obj_relations = []
    prep_obj_relations = []

    # 使用stanfordnlp进行依存句法分析
    doc = nlp(text)
    for sentence in doc.sentences:
        # 存储每个单词的依存关系和对应的编号
        dep_dict = {word.id: {'word': word.text, 'rel': word.deprel, 'head': word.governor} for word in sentence.words}

        # 提取主语-谓语-宾语关系
        for word in sentence.words:
            if word.upos == 'VERB':
                verb_id = word.id
                subject = ''
                obj = ''
                for dep_id, dep in dep_dict.items():
                    if dep['head'] == verb_id and dep['rel'] == 'nsubj':
                        subject = dep['word']
                    elif dep['head'] == verb_id and dep['rel'] == 'obj':
                        obj = dep['word']
                if subject and obj:
                    subj_pred_obj_relations.append((subject, word.text, obj))

        # 提取定语-后置动宾关系
        for word in sentence.words:
            if word.deprel == 'amod' and dep_dict[word.head]['rel'] == 'vobj':
                adj_post_obj_relations.append((word.text + dep_dict[word.head]['word'], dep_dict[word.head]['word']))

        # 提取含有介宾关系的主谓动补关系
        for word in sentence.words:
            if word.deprel == 'comp:prep' and dep_dict[word.head]['upos'] == 'VERB':
                verb = dep_dict[word.head]['word']
                prep_obj = ''
                for dep_id, dep in dep_dict.items():
                    if dep['head'] == word.id and dep['rel'] == 'case':
                        prep_obj += dep['word']
                    elif dep['head'] == word.id and dep['rel'] == 'nmod':
                        prep_obj += dep['word']
                if prep_obj:
                    prep_obj_relations.append((verb, prep_obj))

    return subj_pred_obj_relations, adj_post_obj_relations, prep_obj_relations

# 测试代码
text = '小明爱吃苹果，他喜欢把红色的苹果放在桌子上。'
# subj_pred_obj_relations, adj_post_obj_relations, prep_obj_relations = extract_relations(text)
# print('主谓宾关系：', subj_pred_obj_relations)
# print('定语后置动宾关系：', adj_post_obj_relations)
# print('主谓动补关系：', prep_obj_relations)

# 使用stanfordnlp进行依存句法分析
doc = nlp(text)
print(doc)