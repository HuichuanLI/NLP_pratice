import math

from text_splitter import TextSplitter
from word_dict import WordDict
from word_graph import WordGraph, Node


class WordBasedTokenizer(object):
    """机械切分分词器，unigram"""

    def __init__(self, word_dict):
        """
        持有word_dict对象，里面存储了词库中的词
        """
        self.word_dict = word_dict

    def _find_all_words(self, sentence, sentence_length):
        """
        找到词图中的所有节点

        对于每一个位置：
            将单字添加到词图
            往后扫描，判断子串是否成词，如果是 添加到词图
        """
        graph = WordGraph()
        all_words = {0: [0], sentence_length - 1: [sentence_length - 1]}
        graph.insert_start_word(WordGraph.NODE_S)

        for i in range(1, sentence_length - 1):
            word_end_index_list = [i]  # 初始化终止节点，将单字时的终止节点加进去
            word = sentence[i]
            freq = self.word_dict.freq(sentence[i]) or 1
            graph.insert_start_word(Node(word, self._calculate_weight(freq), freq=freq))  # 将单字加入起始节点

            # 向后扫描，看是否成词
            j = i + 1  # 词尾位置
            fragment = ''.join(sentence[i:j + 1])
            while j < sentence_length - 1:
                source = self.word_dict.is_in(fragment)  # 判断fragment是否在词库（是词 或 词的前缀）
                if not source:  # 不是词或前缀，停止往后扫描
                    break

                freq = self.word_dict.freq(fragment)  # 取出fragment的词频
                if freq:  # 如果是词（而不是前缀），添加此节点到词图
                    word_end_index_list.append(j)
                    graph.insert_start_word(Node(fragment, self._calculate_weight(freq), source, freq))

                # 词尾 + 1，生成新的fragment
                j += 1
                fragment = ''.join(sentence[i:j + 1])

            all_words[i] = word_end_index_list

        return graph, all_words

    def _make_word_graph(self, sentence, sentence_length):
        """
        构建词图

        扫描句子中的所有词，构建节点列表
        对于每个位置，插入终止节点
        """
        # 扫描句子中的所有词，得到词图（仅包含节点列表）
        graph, all_words = self._find_all_words(sentence, sentence_length)

        # 针对每个位置，插入终止节点
        for i in range(sentence_length - 1):
            for end_words in self._find_end_words_list(i, all_words):
                graph.insert_end_words(end_words)

        # 返回图
        return graph

    @staticmethod
    def _find_end_words_list(start_char_index, all_words):
        """
        寻找当前位置的所有节点，每个对应的终止节点
        """
        for end_index in all_words[start_char_index]:  # 对当前位置的所有节点的end_index
            base_index = sum([len(all_words[i]) for i in range(end_index + 1)])  # 当前节点的node序号
            end_words = [base_index + count for count in range(len(all_words[end_index + 1]))]  # 终止节点的node序号
            yield end_words

    def seg(self, sentence):
        """
        分词主函数

        构建词图，计算unigram最优路径，依次返回最优路径上的词
        """
        sentence = [WordGraph.NODE_S.key] + list(sentence) + [WordGraph.NODE_E.key]

        # 构建词图
        graph = self._make_word_graph(sentence, len(sentence))

        # 计算最优路径
        route = graph.calculate()

        # 根据最优路径，依次返回word
        index = route[0][1]  # 初始节点
        node = graph.get_node(index)
        while node != WordGraph.NODE_E:  # 不断往后找最优路径的word，直至终止
            word = node.key
            yield word
            index = route[index][1]
            node = graph.get_node(index)

    def _calculate_weight(self, freq):
        """
        根据词频计算权重

        log p = log (freq/total_freq) = log(freq) - log(total_freq)

        unigram分词：
        p(w1,w2,w3,...) = p(w1)p(w2)p(w3)...
        log p(w1,w2,w3,...) = log p(w1) + log p(w2) + log p(w3) + ...
        故可使用word_graph的加法最优路径
        """
        return math.log(freq or freq + 1) - self.word_dict.get_total_freq_log()


if __name__ == '__main__':

    word_dict = WordDict('core_dict.txt')
    tokenizer = WordBasedTokenizer(word_dict)
    text_splitter = TextSplitter()

    content = '今天天气不错，祝你马到成功'
    words = []
    for sent in text_splitter.split_sentence_for_seg(content):
        if sent in text_splitter.stops[2:-2]:  # 分隔符，直接添加到words，无需构建词图
            words.append(sent)
            continue
        words.extend(list(tokenizer.seg(sent)))
    print(' '.join(words))
