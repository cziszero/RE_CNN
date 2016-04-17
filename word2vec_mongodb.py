# -*- encoding: utf-8 -*-
# Created by Han on 2016/3/8
import pymongo
import datetime

class GloveVecPorcesser(object):
    """
    处理glove中词向量文件中的每一行。
    """
    def __init__(self, wordvec_str):
        """
        初始化
        :param wordvec_str:
         glove中的词向量形式，字符串。
         eg. "the 0.418 0.24968 -0.41242 0.1217 0.34527 -0.044457 -0.49688 -0.17862 ..."
        """
        self.wordvec_str = wordvec_str
        self.word = None
        self.vec = None

    def process(self):
        """
        分别取出字符串中的词和对应向量。
        :return:
        元组，（词_字符串，对应向量_列表）
        """
        all_list = self.wordvec_str.strip().split()
        self.word = all_list[0]
        self.vec = " ".join(all_list[1:])

        return self.word, self.vec


def mongo_wordvec_save(wordvec_file, mongo_collection):
    """
    将词向量文件中的词向量存入到mongodb中。
    :param wordvec_file:
    词向量文件。
    :param mongo_collection:
     要存的mongodb_collection。
    """
    with open(wordvec_file) as file_wordvec:
        vec_counter = 0
        newvec_counter = 0
        wordvec_list =[]
        for wordvec_str in file_wordvec.xreadlines():
            vec_counter += 1
            word, vec = GloveVecPorcesser(wordvec_str).process()
            if not mongo_collection.find_one({"word": word}):
                newvec_counter += 1
                wordvec_list.append({"word": word, "vec": vec})
                if (newvec_counter) % 1000 == 0:
                    mongo_collection.insert(wordvec_list)
                    print "Processed %d Vectors.\t\t%d New Vectors.\t\tTime: %s"\
                          % (vec_counter, newvec_counter, datetime.datetime.now())
                    wordvec_list = []
        if wordvec_list:
            mongo_collection.insert(wordvec_list)
            print "Processed %d Vectors.\t\t\t%d New Vectors.\t\t\tTime: %s" \
                  % (vec_counter, newvec_counter, datetime.datetime.now())


if __name__ == "__main__":
    mongo_client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    mongo_db = mongo_client["wordvec"]
    # mongo_collection = mongo_db["glove840B300d"]
    mongo_collection = mongo_db["glove6B50d"]
    # wordvec_file = "D:\\Work Space\\Data Sets\\WordVec\\GloVe\\glove.840B.300d\\glove.840B.300d.txt"
    wordvec_file = "D:\\Work Space\\Data Sets\\WordVec\\GloVe\\glove.6B\\glove.6B.50d.txt"
    # 将词向量存入mongodb中
    mongo_wordvec_save(wordvec_file, mongo_collection)

    # 将词向量中包含"_"的词放到"dup_glove.txt"中
    # show_info = mongo_collection.find({"word": {"$regex": ".*_.*"}})
    # with open("result\dup_glove.txt", "w") as dup_glove:
    #     for info in show_info:
    #         dup_glove.write(info["word"].encode("utf8") + "\n")
    #     print 'Processing End.'

