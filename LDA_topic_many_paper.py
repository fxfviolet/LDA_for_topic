import lda
import jieba 
import re 
import collections
import numpy as np
import sys 
import os
import common_util
import text_progress

"""
@Desc :  对一个文件中的多篇文本用LDA提取文章的主题词
"""

class lda_keywords_many_paper(object):
    
    util_text = text_progress()
    
    def jieba_cut_many_paper(self,file):
        """
        对一个文件中的每一篇文章分词，返回词语列表和词语字典列表的集合
        """
        rules =  u"([\u4e00-\u9fff]+)"
        pattern =  re.compile(rules)
        docs = {}
        i = 0
        with open(file,'r',encoding='utf-8') as f:   
            for line in f.readlines():        
                docs[i] = {}
                word_list = []         
                line = line.replace("\r","").replace("\\n","").strip()    
                if line != '':           
                    line = ' '.join(jieba.cut(line))
                    seg_list = pattern.findall(line)
                    for word in seg_list:
                        if word not in self.util_text.stopwords:
                            if len(word) > 1:
                                word_list.append(word)
                    word_set = list(set(word_list))
                    docs[i]['word_list'] = word_list
                    docs[i]['word_set'] = word_set
                    i += 1
        return docs
    
    def docs_freq_matrix(self,docs):
        """
        对词语字典根据词频转换成数字矩阵
        """
        all_X = []
        for key,values in docs.items():
            word_matrix = []
            word_dict = collections.Counter(values['word_list'])
            word_key = list(word_dict.keys())
            word_freq = []
            for word in values['word_set']:
                if word in word_key:
                    word_freq.append(word_dict[word])
                else:
                    word_freq.append(0)
            word_matrix.append(word_freq)
            X = np.array(word_matrix)
            all_X.append(X)
        return all_X
    
    def lda_model(self,file,n_topics=10,K=10):
        """
        用lda算法提取主题词
        影响主题词的三个参数：
        alpha:决定每篇文本属于每个topic的概率; eta:决定每个词语在每个topic内的权重; n_iter:迭代次数:决定loglikelihood的收敛
        """
        docs = self.jieba_cut_many_paper(file)
        all_X = self.docs_freq_matrix(docs)   
        all_keywords = {}
        for i,X in enumerate(all_X):
            all_keywords[i] = []
            if X != []:         
                model = lda.LDA(n_topics=n_topics, n_iter=1000,alpha=0.01,eta=10,random_state=1,refresh=200)
                model.fit(X)

                doc_topic = model.doc_topic_ 
                topic_word = model.topic_word_

                topic_most_pr = doc_topic[0].argmax()
                keywords = np.array(docs[i]['word_set'])[np.argsort(topic_word[topic_most_pr])][:-(K+1):-1]
                all_keywords[i].append(keywords)               
        return all_keywords
    
    
if __name__ == '__main__':
    pass
    file = "./文章集合.txt"
    text = lda_keywords_many_paper()
    all_keywords = text.lda_model(file, n_topics=10, K=10)
    print(all_keywords)
