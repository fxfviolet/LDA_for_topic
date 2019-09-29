import codecs
import collections
import numpy as np
import lda
import jieba
import re


class lda_get_keywords(object):

    def read_stop_words(self, file_stopwords):
        stop_words = []
        with open(file_stopwords, "r", encoding="UTF-8") as f:
            for line in f.readlines():
                line = line.strip()
                stop_words.append(line)
        stop_words = set(stop_words)
        return stop_words

    def jieba_cut_words(self, file_origin, file_stopwords):
        stop_words = self.read_stop_words(file_stopwords)
        rules = u"([\u4e00-\u9fff]+)"
        pattern = re.compile(rules)
        with open(file_origin, "r", encoding="UTF-8") as f:
            word_list = []
            lines = f.readlines()
            for line in lines:
                line = line.replace("\r", "").replace("\n", "").strip()
                if line == "" or line is None:
                    continue
                line = ' '.join(jieba.cut(line))
                seg_list = pattern.findall(line)
                for word in seg_list:
                    if word not in stop_words:
                        word_list.append(word)
        word_set = list(set(word_list))
        return word_list, word_set

    def words_freq_matrix(self, word_list, word_set):
        word_matrix = []
        word_dict = collections.Counter(word_list)
        word_key = list(word_dict.keys())
        word_freq = []
        for word in word_set:
            if word in word_key:
                word_freq.append(word_dict[word])
            else:
                word_freq.append(0)
        word_matrix.append(word_freq)
        X = np.array(word_matrix)
        return X

    def lda_model(self, file_origin, file_stopwords, n_topics, n_iter, K):
        word_list, word_set = self.jieba_cut_words(file_origin, file_stopwords)
        X = self.words_freq_matrix(word_list, word_set)

        model = lda.LDA(n_topics=n_topics, n_iter=n_iter, random_state=1)
        model.fit(X)

        doc_topic = model.doc_topic_  # doc_topic: 每篇文本属于每个topic的概率
        topic_word = model.topic_word_  # topic_word：每个topic内词的分布，包含这个词的概率/权重

        topic_most_pr = doc_topic[0].argmax()  # 概率最高的topic
        key_words = np.array(word_set)[np.argsort(topic_word[topic_most_pr])][:-(K + 1):-1]  # 概率最高的topic内权重最高的词语

        return key_words


if __name__ == '__main__':
    pass
    file_origin = "./text.txt"
    file_stopwords = "./stopwords.txt"

    text = lda_get_keywords()
    key_words = text.lda_model(file_origin, file_stopwords, n_topics=5, n_iter=50, K=5)
    print(key_words)
