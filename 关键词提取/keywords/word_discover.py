import math
from collections import Counter
from collections.abc import Iterable
from functools import reduce
from operator import mul

from pygtrie import Trie

from utils.text_splitter import TextSplitter


class WordDiscover(object):
    """
    通过左右熵与互信息实现的新词发现

    本代码主要参考自SmoothNLP:
        https://github.com/smoothnlp/SmoothNLP/blob/master/smoothnlp/algorithm/phrase/ngram_utils.py
    """

    def __init__(self):
        self.text_splitter = TextSplitter()

    @staticmethod
    def union_word_freq(dic1, dic2):
        """
            word_freq合并
        Args:
            dic1: {'你':200,'还':2000,....}
            dic2: {'你':300,'是':1000,....}
        Returns:
            {'你':500,'还':2000,'是':1000,....}
        """
        keys = (dic1.keys()) | (dic2.keys())
        total = {}
        for key in keys:
            total[key] = dic1.get(key, 0) + dic2.get(key, 0)
        return total

    @staticmethod
    def generate_ngram(sentence_chunk: list, n: int = 2):
        """
            对一句话生成n-gram并统计词频字典，n=token_length,
        Args:
            sentence_chunk: list
            n:  n-gram中的n
        Returns:
            generator
        """
        for sentence in sentence_chunk:
            for i in range(0, len(sentence) - n + 1):
                yield sentence[i:i + n]

    @staticmethod
    def get_ngram_freq_info(sentences: iter,
                            min_n: int = 2,
                            max_n: int = 4,
                            chunk_size: int = 100000,
                            min_freq: int = 2,
                            ):

        """
            统计corpus中的词频，与不同n下n-gram的词
        Args:
            sentences:
            min_n: n
            max_n:
            chunk_size:  批处理大小
            min_freq:    统计的词的最小频率
        Returns:
            ngram_freq_total: {'你':500,'还':2000,'是':1000, ...}
            ngram_keys:  {1:{'好','还','是'}, 2:{‘你好'....}, ...}
        """

        ngram_freq_total = {}  # 记录词频
        ngram_keys = {i: set() for i in range(1, max_n + 2)}  # 用来存储N=时, 都有哪些词

        def _process_corpus_chunk(sentence_chunk: list):
            ngram_freq = {}
            for ni in [1] + list(range(min_n, max_n + 2)):
                ngram_generator = WordDiscover.generate_ngram(sentence_chunk, ni)
                nigram_freq = dict(Counter(ngram_generator))
                ngram_keys[ni] = (ngram_keys[ni] | nigram_freq.keys())
                ngram_freq = {**nigram_freq, **ngram_freq}
            ngram_freq = {word: count for word, count in ngram_freq.items() if
                          count >= min_freq}  # 每个chunk的ngram频率统计
            return ngram_freq

        # 批处理
        sentence_chunk = []
        for sentence in sentences:
            if len(sentence_chunk) < chunk_size:
                sentence_chunk.append(sentence)
            else:

                ngram_freq = _process_corpus_chunk(sentence_chunk)
                ngram_freq_total = WordDiscover.union_word_freq(ngram_freq, ngram_freq_total)
                sentence_chunk = []
        # 最后一批的处理
        ngram_freq = _process_corpus_chunk(sentence_chunk)
        ngram_freq_total = WordDiscover.union_word_freq(ngram_freq, ngram_freq_total)

        for k in ngram_keys:
            ngram_keys[k] = ngram_keys[k] & ngram_freq_total.keys()
        return ngram_freq_total, ngram_keys

    @staticmethod
    def _ngram_entropy_score(parent_ngrams_freq):
        """
        根据邻字neighbor的出现频率，按照公式，计算一个词左熵或右熵

        Args:
            parent_ngrams_freq: 该的的父级词的频率统计 [5,6,7..]
        """
        total_freq = sum(parent_ngrams_freq)  # 总词频数，即当前gram的词频
        parent_ngram_probas = [
            freq / total_freq for freq in parent_ngrams_freq  # P(W_neighbor|W) = Count(W,W_neighbor) / Count(W)
        ]
        entropy = sum([-1 * prob * math.log(prob, 2) for prob in parent_ngram_probas])
        return entropy

    @staticmethod
    def _calc_ngram_entropy(ngram_freq, ngram_keys, n: list):
        """
        基于ngram频率信息计算左右熵信息
        Args:
            ngram_freq:
            ngram_keys:
            n:
        Returns:
            {'word':(left_entropy, right entropy), ....}
        """

        if isinstance(n, Iterable):  # 一次性计算 len(N)>1 的 ngram
            entropy = {}
            for ni in n:
                entropy = {**entropy, **WordDiscover._calc_ngram_entropy(ngram_freq, ngram_keys, ni)}
            return entropy
        ngram_entropy = {}
        target_ngrams = ngram_keys[n]
        parent_candidates = ngram_keys[n + 1]

        # 对 n+1 gram 进行建Trie，便于快速查找到n-gram的所有 左邻字 和 右邻字
        # "开宝马"，将"开宝马"加到右邻trie树，将"宝马开"加到左邻trie树
        # 便于以后：遇到"开宝"，从右邻trie树中取到"马"的右邻；遇到"宝马"，从左邻trie树中取到"开"的左邻
        left_neighbors = Trie()
        right_neighbors = Trie()
        for parent_candidate in parent_candidates:
            right_neighbors[parent_candidate] = ngram_freq[parent_candidate]
            left_neighbors[parent_candidate[1:] + parent_candidate[0]] = ngram_freq[parent_candidate]

        # 计算
        for target_ngram in target_ngrams:
            try:  # 一定情况下, 一个candidate ngram 没有左右neighbor
                right_neighbor_counts = (right_neighbors.values(target_ngram))
                right_entropy = WordDiscover._ngram_entropy_score(right_neighbor_counts)

            except KeyError:
                right_entropy = 0
            try:
                left_neighbor_counts = (left_neighbors.values(target_ngram))

                left_entropy = WordDiscover._ngram_entropy_score(left_neighbor_counts)
            except KeyError:
                left_entropy = 0
            ngram_entropy[target_ngram] = (left_entropy, right_entropy)

        return ngram_entropy

    @staticmethod
    def _calc_ngram_pmi(ngram_freq, ngram_keys, n):
        """
        计算 Pointwise Mutual Information 与 Average Mutual Information

        Args:
            ngram_freq: dict 词频
            ngram_keys: dict
            n: []

        Returns:
            { 'word': (PMI, APMI)}
        """

        if isinstance(n, Iterable):
            mi = {}
            for ni in n:
                mi = {**mi, **WordDiscover._calc_ngram_pmi(ngram_freq, ngram_keys, ni)}
            return mi
        n1_total_count = sum([ngram_freq[k] for k in ngram_keys[1] if k in ngram_freq])
        target_n_total_count = sum([ngram_freq[k] for k in ngram_keys[n] if k in ngram_freq])
        mi = {}
        for target_ngram in ngram_keys[n]:
            # 联合概率
            joint_proba = ngram_freq[target_ngram] / target_n_total_count
            # 边缘概率相乘
            indep_proba = reduce(mul, [ngram_freq[char] for char in target_ngram]) / (n1_total_count ** n)

            pmi = math.log(joint_proba / indep_proba, 2)  # 互信息
            ami = pmi / len(target_ngram)  # 平均互信息
            mi[target_ngram] = (pmi, ami)
        return mi

    @staticmethod
    def get_scores(sentences, min_n=2, max_n=4, chunk_size=100000, min_freq=1):
        """
            基于corpus, 计算所有候选词汇的相关评分.
        Args:
            sentences: 句子序列
            min_n: 生词的最大gram
            max_n: 生词的最小gram
            chunk_size: 批处理大小
            min_freq: 词最小的频率, 由于跟计算左右熵有关，频率越低，生成算法越准确，建议设置为1, 语料库较大时，可提升至5-10
        Returns:
            with score:     [(word, score), (word, score)]
            without score:  [word, word, ...]
            score越大，生词的可信度越高，默认按分数降序。
        """

        # 先统计词频和ngram对应的词典
        # 后面计算左右熵和互信息都需要使用这两个dict
        ngram_freq, ngram_keys = WordDiscover.get_ngram_freq_info(sentences, min_n, max_n, chunk_size,
                                                                  min_freq=min_freq)
        # 计算词的left_right_entropy: {'word':(left_entropy, right entropy)}
        left_right_entropy = WordDiscover._calc_ngram_entropy(ngram_freq, ngram_keys, list(range(min_n, max_n + 1)))

        # 计算词的 PMI（点积互信息）和APMI（平均互信息）
        mi = WordDiscover._calc_ngram_pmi(ngram_freq, ngram_keys, range(min_n, max_n + 1))

        def word_liberalization(l_entropy, r_entropy):
            """
            利用left entropy right entropy 求词的自由度
            Args:
                l_entropy: left  entropy
                r_entropy: right entropy
            Returns:
                word_liberalization
            """
            if l_entropy == 0 or r_entropy == 0:
                return 0
            return math.log(
                (l_entropy * 2 ** r_entropy + r_entropy * 2 ** l_entropy + 0.00001) / (abs(l_entropy - r_entropy) + 1),
                1.5)

        # # 算法至此结束， 统计每个词的分数
        joint_phrase = mi.keys() & left_right_entropy.keys()  # 求词的交集
        word_info_scores = {
            word: (mi[word][0],  # mutual information
                   mi[word][1],  # average mutual information
                   left_right_entropy[word][0],  # left_entropy
                   left_right_entropy[word][1],  # right_entropy
                   # branch entropy  BE=min{l_entropy,r_entropy}
                   min(left_right_entropy[word][0], left_right_entropy[word][1]),
                   word_liberalization(  # our score
                       left_right_entropy[word][0], left_right_entropy[word][1]) + mi[word][1]
                   ) for word in joint_phrase
        }

        # 对在candidate ngram中, 首字或者尾字出现次数特别多的进行筛选, 如"XX的,美丽的,漂亮的"剔出字典
        target_ngrams = word_info_scores.keys()
        start_chars = Counter([n[0] for n in target_ngrams])
        end_chars = Counter([n[-1] for n in target_ngrams])
        # 设定阈值
        threshold = int(len(target_ngrams) * 0.004)
        threshold = max(50, threshold)
        invalid_start_chars = set([char for char, count in start_chars.items() if count > threshold])
        invalid_end_chars = set([char for char, count in end_chars.items() if count > threshold])
        invalid_target_ngrams = set(
            [n for n in target_ngrams if (n[0] in invalid_start_chars or n[-1] in invalid_end_chars)]
        )
        for n in invalid_target_ngrams:  # 按照不合适的字头字尾信息删除一些
            word_info_scores.pop(n)

        return word_info_scores

    def discover(self, texts, top_k=200, min_gram=2, max_gram=6, min_freq=2, chunk_size=100000,
                 with_score=True, min_score=0.0):

        """
        新词发现算法入口

        Args:
            chunk_size: 批处理统计词频的chunk size
        Returns:
            list: [ word1, word2 ....]  降序
        """
        sentences = (
            sent for text in texts for sent in self.text_splitter.split_sentence_for_seg(text) if len(sent) > 2
        )

        # 基于ngram计算左右熵, 凝聚度
        word_info_scores = WordDiscover.get_scores(sentences, min_gram, max_gram, chunk_size, min_freq)

        # 排序，按照min_score过滤，取top_k
        new_words = []
        for word, (_, _, _, _, _, score) in sorted(  # 按照score排序，倒序
                word_info_scores.items(), key=lambda item: item[1][-1], reverse=True
        ):
            if score <= min_score:
                continue
            new_words.append((word, score) if with_score else word)

        return new_words[:top_k]
