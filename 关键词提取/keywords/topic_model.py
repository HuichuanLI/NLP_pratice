import os

import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel


class TopicModel(object):
    def __init__(self):
        self.lda_model = None
        self.lda_dict = None

    def train_lda_model(self, seg_files, output_model_dir, splitter=' ', num_topics=100):
        # 迭代所有分好词的文件，装载到cut_documents
        cut_documents = []
        for seg_file in seg_files:
            with open(seg_file, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cut_documents.append(line.split(splitter))

        # 构建lda dict, 构建lda model
        self.lda_dict = Dictionary(cut_documents)
        corpus = [self.lda_dict.doc2bow(document) for document in cut_documents]
        # 设置语料库、主题数、lda dict、训练轮数（passe等价于epoch）
        self.lda_model = LdaModel(corpus, num_topics=num_topics, id2word=self.lda_dict, passes=20)

        # 保存模型
        if not os.path.exists(output_model_dir):
            os.mkdir(output_model_dir)
        self.lda_model.save('{}/lda_model'.format(output_model_dir))
        self.lda_dict.save('{}/dictionary'.format(output_model_dir))

    def load_lda_model(self, model_dir):
        self.lda_model = LdaModel.load(model_dir + '/lda_model')
        self.lda_dict = Dictionary.load(model_dir + '/dictionary')

    def extract_keywords(self, words):
        assert self.lda_model and self.lda_dict, "请确保lda模型被加载"

        # 获取文档的主题分布
        doc_topics_distribute = []
        if words:
            doc_topics_distribute = [0] * self.lda_model.num_topics  # 全零初始化主题向量
            bow_content = self.lda_dict.doc2bow([word for word in words])  # 获取词袋id形式的当前文档
            for topic_idx, prob in self.lda_model.get_document_topics(bow_content, minimum_probability=0):  # 迭代文档主题分布
                doc_topics_distribute[topic_idx] = prob  # 更新到doc_topics_distribute

        # 获取词语的主题分布
        new_words = []  # 保存words中，在词库中出现的词
        # 文档主题分布矩阵，保存所有词的主题分布，shape为 len(words) * topic_num，
        # dim 0与new_words一一对应
        words_topics_distribute = []
        for word in set(words):
            if word not in self.lda_dict.token2id:  # 忽略词库外的word
                continue
            new_words.append(word)
            word_topics_distribute = [0] * self.lda_model.num_topics  # 全零初始化主题向量
            for topic_idx, prob in self.lda_model.get_term_topics(  # 迭代词的主题分布
                    self.lda_dict.token2id[word], minimum_probability=0
            ):
                word_topics_distribute[topic_idx] = prob  # 更新到word_topics_distribute
            words_topics_distribute.append(word_topics_distribute)  # 矩阵添加一行

        # 计算相似度，作为词的权重
        weights = []
        if doc_topics_distribute and words_topics_distribute:
            weights = list(self._cosine_similarity([doc_topics_distribute], words_topics_distribute)[0])

        tags = list(zip(new_words, weights))
        return sorted(tags, key=lambda item: item[1], reverse=True)

    @staticmethod
    def _cosine_similarity(matrix_a, matrix_b):
        """计算两个矩阵的余弦相似度"""
        normed_matrix_a = np.array(matrix_a) / np.linalg.norm(matrix_a, ord=2, axis=1)[:, np.newaxis]
        normed_matrix_b = np.array(matrix_b) / np.linalg.norm(matrix_b, ord=2, axis=1)[:, np.newaxis]
        simi = np.dot(normed_matrix_a, normed_matrix_b.T)
        simi[np.isnan(simi)] = 0
        return simi
