# -*- encoding: utf-8 -*-
# Created by Han on 2016/2/26
import theano
import numpy as np
import pandas as pd
import re
import pymongo
import pickle
import time
from nltk.corpus import wordnet as wn


class SentenceProcesser(object):
    """
    句子处理类，处理句子字符串。
    """

    def __init__(self, input_sentence):
        """
        :param input_sentence:
         需要处理的句子
        :return:
        """
        self.sentence = input_sentence
        self.sentence_cleared = None
        self.sentence_list = None
        self.entity_pos = None

    def clear_sentence(self, TREC=False, PUNCT=True):
        """
        清洗输入的句子，去掉后续不需要使用的符号
        :param TREC:
        TREC为True时，保留原大小写，默认False统一使用小写
        :param PUNCT:
         PUNCT为True时保留标点，否则去除标点
        :return:
        """
        # 删除句子起始结尾处的编号、引号和换行符
        # sentence_clearing = re.sub(r"^[0-9]{0,}\t\"", "", sentence)
        sentence_clearing = re.sub(r"^.*?\"", "", self.sentence)
        sentence_clearing = re.sub(r"\"\n", "", sentence_clearing)
        # 处理</e1>为<ende1>形式，因为句子中有a/b的句子形式，而且在无需标点的情况下，去除标点的过程中也会对此处造成影响
        sentence_clearing = re.sub(r"<\/", r"<end", sentence_clearing)
        if PUNCT:
            # 删除句子中不常见的符号，这里把连接符也删掉
            sentence_clearing = re.sub(r"[^A-Za-z0-9(),\.!?<>\'\`]", " ", sentence_clearing)
            # 保留标点，对句子中的标点符号进行处理
            sentence_clearing = re.sub(r",", " , ", sentence_clearing)
            sentence_clearing = re.sub(r"\.", " . ", sentence_clearing)
            sentence_clearing = re.sub(r"!", " ! ", sentence_clearing)
            sentence_clearing = re.sub(r"\(", " \( ", sentence_clearing)
            sentence_clearing = re.sub(r"\)", " \) ", sentence_clearing)
            sentence_clearing = re.sub(r"\?", " \? ", sentence_clearing)
        else:
            # 删除标点
            sentence_clearing = re.sub(r"[^A-Za-z0-9<>\'\`]", " ", sentence_clearing)
        # 对如下引用形式进行处理"'the mountain'"
        pattern_quote = re.compile(r" \'[^\']+\'[ \.]")
        match_quote_list = pattern_quote.findall(sentence_clearing)
        for match_quote in match_quote_list:
            if len(match_quote) > 4:
                match_changed = "' " + match_quote[2:-2] + " '"
                sentence_clearing = sentence_clearing.replace(match_quote[1:-1], match_changed)

        # 对句子中的英文缩写进行处理
        sentence_clearing = re.sub(r"\'s", " \'s", sentence_clearing)
        sentence_clearing = re.sub(r"\'ve", " \'ve", sentence_clearing)
        sentence_clearing = re.sub(r"n\'t", " n\'t", sentence_clearing)
        sentence_clearing = re.sub(r"\'re", " \'re", sentence_clearing)
        sentence_clearing = re.sub(r"\'d", " \'d", sentence_clearing)
        sentence_clearing = re.sub(r"\'ll", " \'ll", sentence_clearing)

        # 对<ende1>[^ ]形式进行处理
        pattern_ee = re.compile(r"<ende\d>[^ ]+?")
        error_end_list = pattern_ee.findall(sentence_clearing)
        for error_end in error_end_list:
            sentence_clearing = sentence_clearing.replace(error_end, error_end.replace(">", "> "))

        # 对[^ ]<e1>形式进行处理
        pattern_es = re.compile("[^ ]+?<e\d>")
        error_start_list = pattern_es.findall(sentence_clearing)
        for error_start in error_start_list:
            sentence_clearing = sentence_clearing.replace(error_start, error_start.replace("<", " <"))

        # 句子中的空白部分替换为1个空格
        sentence_cleared = re.sub(r"\s{2,}", " ", sentence_clearing)
        self.sentence_cleared = sentence_cleared.strip() if TREC else sentence_cleared.strip().lower()

    def get_list_pos(self):
        """
        将清洗后的句子转换为list，list中每项对应后续的一个词向量。
        :return:
        """
        pattern_e1 = re.compile(r"<e1>.*<ende1>")
        pattern_e2 = re.compile(r"<e2>.*<ende2>")
        entity1 = pattern_e1.findall(self.sentence_cleared)[0]
        entity2 = pattern_e2.findall(self.sentence_cleared)[0]
        entity1_info = re.sub(r"<.*?>", "", entity1.strip().replace(" ", "_"))
        entity2_info = re.sub(r"<.*?>", "", entity2.strip().replace(" ", "_"))
        sentence_return = self.sentence_cleared.replace(entity1, entity1_info)
        sentence_return = sentence_return.replace(entity2, entity2_info)
        self.sentence_list = sentence_return.split()

        # 这里获取位置时未使用上面的entity_info是为了防止其他位置有相同的内容
        tmp = re.sub(pattern_e1, "<e1>", self.sentence_cleared)
        sentence_entity_tag = re.sub(pattern_e2, "<e2>", tmp)
        sentence_list_e = sentence_entity_tag.split()
        pos_e1 = sentence_list_e.index("<e1>")
        pos_e2 = sentence_list_e.index("<e2>")
        self.entity_pos = (pos_e1, pos_e2)

    def process(self):
        """
        对原始数据中的句子进行处理，如果不成功则打印句子。
        :return:
        成功处理返回元组，第一项为句子转化成的列表，第二项为实体位置的元组。
        失败则返回False。
        """
        try:
            self.clear_sentence()
            self.get_list_pos()
        except:
            print("Sentence:", self.sentence)
            return False
        else:
            return self.sentence_list, self.entity_pos


def voc_set_add(sentence_word_list, vocabulary_set):
    """
    将句子中的词加入到词表中。
    :param sentence_word_list:
    列表形式的句子。
    :param vocabulary_set:
    词表。
    :return:
    添加后的词表。
    """
    for word in sentence_word_list:
        vocabulary_set.add(word)
    return vocabulary_set


def rel_set_add(relationship, relation_set):
    """
    将关系字符串添加到关系词典中。
    :param relationship:
    关系字符串。
    :param relation_set:
    关系集合。
    :return:
    关系结合
    """
    relation_set.add(relationship)
    return relation_set


def get_lr(sentence, pos):
    """
    得到句子列表中pos位置左右的词，如果
    :param sentence:
    句子列表。
    :param pos:
    要获取左右信息的词的位置。
    :return:
    左右的词。
    """
    if pos == 0 and pos != len(sentence)-1:
        lw = "\\s"
        rw = sentence[pos+1]
    elif pos == len(sentence)-1 and pos != 0:
        lw = sentence[pos-1]
        rw = "\\e"
    else:
        lw = sentence[pos-1]
        rw = sentence[pos+1]
    return lw, rw


def lex_feature(cleared_data):
    """
    获取数据的词汇特征
    :param cleared_data:
    :type pandas.DataFrame:
    清洗后的数据。
    :return:
    """
    L1_list, L2_list, L3_list, L4_list, L5_list = [], [], [], [], []
    for i in range(len(cleared_data)):
        e1 = cleared_data.ix[i, "Sentences"][cleared_data.ix[i, "EntitiesPos"][0]]
        e2 = cleared_data.ix[i, "Sentences"][cleared_data.ix[i, "EntitiesPos"][1]]
        L1_list.append(e1)
        L2_list.append(e2)
        L3_list.append(get_lr(cleared_data.ix[i, "Sentences"], cleared_data.ix[i, "EntitiesPos"][0]))
        L4_list.append(get_lr(cleared_data.ix[i, "Sentences"], cleared_data.ix[i, "EntitiesPos"][1]))
        wn_e1 = wn.synsets(e1)[0] if wn.synsets(e1) else None
        wn_e2 = wn.synsets(e2)[0] if wn.synsets(e2) else None
        hype_e1 = (wn_e1.hypernyms()[0].name().split(".")[0] if wn_e1.hypernyms() else "nohype") if wn_e1 else "nohype"
        hype_e2 = (wn_e2.hypernyms()[0].name().split(".")[0] if wn_e2.hypernyms() else "nohype") if wn_e2 else "nohype"
        L5_list.append((hype_e1.encode('utf8'), hype_e2.encode('utf8')))
    return L1_list, L2_list, L3_list, L4_list, L5_list


def data_clear(input_file):
    """
    将原始数据文件转换为句子词列表和关系字符串。
    :param input_file:
    原始数据。
    :return:
    """
    vocabulary_set = set()
    relation_set = set()
    with open(input_file, "r") as data_file:
        data_content = data_file.readlines()
        cleared_data = []
        for i in range(len(data_content) / 4):
            sentence_word_list, entity_pos = SentenceProcesser(data_content[4 * i]).process()
            vocabulary_set = voc_set_add(sentence_word_list, vocabulary_set)
            relationship = data_content[4 * i + 1].replace('\n', '')
            relation_set = rel_set_add(relationship, relation_set)
            cleared_data.append([sentence_word_list, entity_pos, relationship])
        cleared_data = pd.DataFrame(cleared_data, columns=['Sentences', 'EntitiesPos', 'Relations'])

    # 词典特征
    L1_list, L2_list, L3_list, L4_list, L5_list = lex_feature(cleared_data)
    cleared_data["L1"] = L1_list
    cleared_data["L2"] = L2_list
    cleared_data["L3"] = L3_list
    cleared_data["L4"] = L4_list
    cleared_data["L5"] = L5_list

    # 将上位词加入词典
    for hypes in L5_list:
        for hype in hypes:
            vocabulary_set.add(hype)

    # 设置句子起始符号为"\s"，结束符为"\e"
    vocabulary_set.add("\\s")
    vocabulary_set.add("\\e")

    return cleared_data, sorted(list(vocabulary_set)), sorted(list(relation_set))


def wordvec_dict_gender(vocab_list, mongo_collection, wordvec_dimension):
    """
    生成词汇集合中对应的词向量，之所以不用Mongodb中的是因为效率问题。
    :param vocab_list:
    词汇列表。
    :param mongo_collection:
    GloVe词向量对应的mongodb collection。
    :param wordvec_dimension:
    词向量长度。
    :return:
    字典：{"词（str）": "向量（numpy.narray, dtype=np.float32"}。
    """
    wordvec_dict = dict()
    for word in vocab_list:
        found_word = mongo_collection.find_one({"word": word})
        if not found_word:
            vector = np.random.uniform(-0.25, 0.25, wordvec_dimension)
        else:
            vector = np.array(found_word['vec'].split(), dtype=np.float32)
        wordvec_dict.setdefault(word, vector)
    return wordvec_dict


def word_index_matrix_gen(vocabulary_list, wordvec_dict, wordvec_dim, offset=0):
    """
    按照词汇列表顺序生成词向量矩阵。
    :param vocabulary_list:
    词汇列表。
    :param wordvec_dict:
    词汇列表中每个词对应的向量。
    :param wordvec_dim:
    词向量维度。
    :return:
    """
    word_index_matrix = np.zeros((len(vocabulary_list)+offset, wordvec_dim), dtype=np.float32)
    for i in range(len(vocabulary_list)):
        word_index_matrix[i+offset] = wordvec_dict[vocabulary_list[i]]
    return word_index_matrix


def relation_dict_gender(rel_list):
    """
    关系对应的数值字典。
    :param rel_list:
    关系列表。
    :return:
    关系向量字典。
    """
    rel_dict = dict()
    # 使用数字标签
    for i in range(len(rel_list)):
        rel_dict[rel_list[i]] = i
    return rel_dict


def posvec_dict_gender(dimension, start_num, end_num):
    """
    生成位置信息对应的向量。
    :param dimension:
    生成向量的维度。
    :return:
    """
    posvec_dict = dict()
    for i in range(start_num, end_num+1):
        posvec_dict.setdefault(i, np.random.uniform(-0.25, 0.25, dimension))
    # 100对应0向量
    posvec_dict.setdefault(end_num + 1, np.zeros(dimension))
    return posvec_dict


def pos_index_matrix_gen(start_num, end_num, posvec_dict, posvec_dim):
    """
    生成位置信息对应的向量矩阵。
    """
    pos_num = end_num - start_num + 2
    pos_index_matrix = np.zeros((pos_num, posvec_dim), dtype=theano.config.floatX)
    for i in range(start_num, end_num+2):
        pos_index_matrix[i-start_num] = posvec_dict[i]
    return pos_index_matrix


def list_to_dict(input_list, offset):
    """
    列表中的值作为key,index作为value，offset为index偏移量
    """
    res_dict = dict()
    for idx in range(len(input_list)):
        res_dict.setdefault(input_list[idx], idx+offset)
    return res_dict


def data_process(data_cleared, voc_dict, relvec_dict, aligned=False, start_num=-99, end_num=99):
    """
    将cleared_data转换为数字形式。
    """
    sents_list = list()
    pos_list = list()
    rel_list = list()
    l1_list = list()
    l2_list = list()
    l3_list = list()
    l4_list = list()
    l5_list = list()
    data_processed = dict()

    max_length = max([len(i) for i in data_cleared["Sentences"]])

    for i in range(len(data_cleared)):
        if i % 100 == 0:
            print("Processed:\t" + str(i) + "\t" + time.ctime())

        sent = data_cleared.ix[i, "Sentences"]
        e1_pos, e2_pos = data_cleared.ix[i, "EntitiesPos"]

        sent_length = len(sent)
        length = max_length if aligned else sent_length

        sent_list = list()
        ep_list = list()

        for word_index in range(length):
            # 处理句子中的词
            if word_index < sent_length:
                sent_list.append(voc_dict[sent[word_index]])
                ep_list.append((word_index - e1_pos - start_num, word_index - e2_pos - start_num))
            else:
                zero_pos_index = end_num - start_num + 1
                sent_list.append(0)     #
                ep_list.append((zero_pos_index, zero_pos_index))    # 位置矩阵第200行对应的向量为0

        sents_list.append(sent_list)
        pos_list.append(ep_list)

        # 3、将关系对应到相应的数字
        rel_list.append(relvec_dict[data_cleared.ix[i, "Relations"]])

        # 4、L1值处理
        l1_list.append(voc_dict[data_cleared.ix[i, "L1"]])

        # 5、L2值处理
        l2_list.append(voc_dict[data_cleared.ix[i, "L2"]])

        # 6、L3值处理
        l3_list.append((voc_dict[data_cleared.ix[i, "L3"][0]], voc_dict[data_cleared.ix[i, "L3"][1]]))

        # 7、L4值处理
        l4_list.append((voc_dict[data_cleared.ix[i, "L4"][0]], voc_dict[data_cleared.ix[i, "L4"][1]]))

        # 8、L5值处理
        l5_list.append((voc_dict[data_cleared.ix[i, "L5"][0]], voc_dict[data_cleared.ix[i, "L5"][1]]))

    data_processed["Sentences"] = sents_list
    data_processed["EntitiesPos"] = pos_list
    data_processed["Relations"] = rel_list
    data_processed["L1"] = l1_list
    data_processed["L2"] = l2_list
    data_processed["L3"] = l3_list
    data_processed["L4"] = l4_list
    data_processed["L5"] = l5_list

    data_processed = pd.DataFrame(data_processed)

    return data_processed


if __name__ == "__main__":
    # 初始化
    PROCESS = False   # Processed为True时重新处理数据，否则读取之前处理好的结果
    GENDATA = True
    ALIGN = True

    # 可以视为配置信息
    wordvec_dimension = 50
    posvec_dimension = 5
    start_num = -99     # 位置信息最小相对位置
    end_num = 99        # 位置信息最大相对位置
    data_file = "dataset\\SemEval2010_task8_all_data\\all_data.txt"
    data_cleared_file = "data_processed\\data_cleared.pkl"
    data_processed_file = "data_processed\\data_processed.pkl"
    data_aligned_file = "data_processed\\data_aligned.pkl"
    list_vocabulary_file = "tmp_files\\list_vocabulary.pkl"
    list_relation_file = "tmp_files\\list_relation.pkl"
    dict_vocabulary_file = "tmp_files\\dict_vocabulary.pkl"
    dict_relation_file = "tmp_files\\dict_relation.pkl"
    dict_wordvec_file = "tmp_files\\dict_wordvec.pkl"
    dict_posvec_file = "tmp_files\\dict_posvec.pkl"
    matrix_word_index_file = "tmp_files\\matrix_wordidx.pkl"
    matrix_pos_index_file = "tmp_files\\matrix_posidx.pkl"

    # 数据处理部分
    if PROCESS:
        # 处理原始数据
        print("Processing data..." + "\t"*3 + time.ctime())

        # 数据清洗
        print("Cleaning data..." + "\t"*3 + time.ctime())
        data_cleared, list_vocab, list_rel = data_clear(data_file)
        pickle.dump(data_cleared, open(data_cleared_file, "w"))
        pickle.dump(list_vocab, open(list_vocabulary_file, "w"))
        pickle.dump(list_rel, open(list_relation_file, "w"))
        print("Data cleared." + "\t"*3 + time.ctime())

        # 词字典生成
        # 生成词对应位置的词典，注意这里要给补零的留出位置。
        print("Generating vocabulary dict..." + "\t"*3 + time.ctime())
        dict_vocabulary = list_to_dict(list_vocab, offset=1)
        pickle.dump(dict_vocabulary, open(dict_vocabulary_file, "w"))
        print("Vocabulary dict generated." + "\t"*3 + time.ctime())

        # 生成位置信息向量字典
        print("Generating posvec dict..." + "\t"*3 + time.ctime())
        dict_posvec = posvec_dict_gender(posvec_dimension, start_num=start_num, end_num=end_num)
        pickle.dump(dict_posvec, open(dict_posvec_file, "w"))
        print("Posvec dict generated." + "\t"*3 + time.ctime())

        # 生成位置信息向量矩阵
        print("Generating posvec matrix..." + "\t"*3 + time.ctime())
        posidx_matrix = pos_index_matrix_gen(
          start_num=start_num, end_num=end_num, posvec_dict=dict_posvec, posvec_dim=posvec_dimension
        )
        pickle.dump(posidx_matrix, open(matrix_pos_index_file, "w"))
        print("Posvec matrix generated." + "\t"*3 + time.ctime())

        # 生成词向量词典
        print("Generating wordvec dict..." + "\t"*3 + time.ctime())
        mongo_client = pymongo.MongoClient(host="127.0.0.1", port=27017)
        mongo_db = mongo_client["wordvec"]
        if wordvec_dimension == 50:
            mongo_collection = mongo_db["glove6B50d"]
        elif wordvec_dimension == 300:
            mongo_collection = mongo_db["glove240B300d"]
        else:
            raise ValueError
        dict_wordvec = wordvec_dict_gender(list_vocab, mongo_collection, wordvec_dimension)
        pickle.dump(dict_wordvec, open(dict_wordvec_file, "w"))
        print("Wordvec dict generated." + "\t"*3 + time.ctime())

        # 生成词向量矩阵
        print("Generating wordvec matrix..." + "\t"*3 + time.ctime())
        matrix_wordidx = word_index_matrix_gen(list_vocab, dict_wordvec, wordvec_dimension, offset=1)
        pickle.dump(matrix_wordidx, open(matrix_word_index_file, "w"))
        print("Wordvec matrix generated." + "\t"*3 + time.ctime())

        # 生成关系向量字典
        print("Generating relvec dict..." + "\t"*3 + time.ctime())
        dict_relation = relation_dict_gender(list_rel)
        pickle.dump(dict_relation, open(dict_relation_file, "w"))
        print("Relvec dict generated." + "\t"*3 + time.ctime())

        print("Processing Finshed." + "\t"*3 + time.ctime())

    else:
        # 直接调用之前处理的结果
        print("Loading data..." + "\t"*3 + time.ctime())
        data_cleared = pickle.load(open(data_cleared_file))
        list_vocab = pickle.load(open(list_vocabulary_file))
        list_rel = pickle.load(open(list_relation_file))
        dict_vocabulary = pickle.load(open(dict_vocabulary_file))
        dict_wordvec = pickle.load(open(dict_wordvec_file))
        dict_relation = pickle.load(open(dict_relation_file))
        matrix_wordidx = pickle.load(open(matrix_word_index_file))
        matrix_posidx= pickle.load(open(matrix_pos_index_file))
        dict_posvec = pickle.load(open(dict_posvec_file))
        print("Loading Finished." + "\t"*3 + time.ctime())

    if GENDATA:
        print("Generating  data..." + "\t"*3 + time.ctime())

        print("Processing cleared data..." + "\t"*3 + time.ctime())
        processed_data = data_process(data_cleared, dict_vocabulary, dict_relation, ALIGN, start_num=start_num, end_num=end_num)
        if ALIGN:
            print("Saveing alignd data..." + "\t"*3 + time.ctime())
            pickle.dump(processed_data, open(data_aligned_file, "w"))
        else:
            print("Saveing processed data..." + "\t"*3 + time.ctime())
            pickle.dump(processed_data, open(data_processed_file, "w"))
        print("Data has been processed." + "\t"*3 + time.ctime())

        print("Generating Finished." + "\t"*3 + time.ctime())

