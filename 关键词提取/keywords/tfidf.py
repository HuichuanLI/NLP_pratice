import math


class TFIDF(object):
    """
    用法：
    1. 离线基于分好词的大规模预料，训练idf；
    2. 加载idf；
    3. 在线计算tf-idf；
    """

    def __init__(self):
        self.idf = {}  # 存放词及其idf权重
        self.idf_median = 0  # 存放idf中位数，防止在线时遇到一些（训练时未见过的）新词

    def compute_tfidf(self, words):
        """
        在线计算tfidf
        Args:
            words: 分好词的一短文本
        Returns:
            list [('word1', weight1), ('word2', weight2), ...]
        """
        # 确保idf已经被加载：idf不为空，并且idf中位数不为0
        assert self.idf and self.idf_median, "请确保idf被加载！"

        # 统计tf
        tf = {}
        for word in words:
            tf[word] = tf.get(word, 0) + 1

        # 从加载好的idf字典中取idf，计算tfidf
        tfidf = {}
        for word in set(words):
            tfidf[word] = tf[word] / len(words) * self.idf.get(word, self.idf_median)

        # 对所有词的tfidf排序，按照权重从高到低排序，返回
        tfidf = sorted(tfidf.items(), key=lambda item: item[1], reverse=True)
        return tfidf

    def load_idf(self, idf_path, splitter=' '):
        """
        加载idf
        """
        with open(idf_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or len(line.split(splitter)) != 2:  # 跳过为空的文本、不合法的文本
                    continue
                term, idf = line.split(splitter)
                self.idf[term] = float(idf)
        self.idf_median = sorted(self.idf.values())[len(self.idf) // 2]  # 计算idf的中位数

    def train_idf(self, seg_files, output_file_name, splitter=' '):
        """
        离线训练idf
        Args:
            seg_files: list 分过词的训练文件列表，txt格式，文档用\n换行
            output_file_name: 输出的标准idf文本，txt格式
            splitter: term之间分隔符，word和idf的分隔符
        """
        # 总文档数初始化为0
        doc_count = 0

        # 统计df
        for seg_file in seg_files:  # 迭代所有文件
            with open(seg_file, encoding='utf-8') as f:
                for line in f:  # 迭代每一行
                    line = line.strip()
                    if not line:
                        continue
                    doc_count += 1  # 更新总文档数
                    words = set(line.split(splitter))
                    for word in words:
                        self.idf[word] = self.idf.get(word, 0) + 1  # 更新当前word的文档频数

        # 计算idf，保存到文件
        with open(output_file_name, 'w', encoding='utf-8') as f:
            for word, df in self.idf.items():
                self.idf[word] = math.log(doc_count / (df + 1))  # 计算idf
                f.write('{}{}{}\n'.format(word, splitter, self.idf[word]))
