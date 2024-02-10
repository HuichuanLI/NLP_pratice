import json
import math


class WordDict(object):
    """词典加载"""

    def __init__(self, dict_path, separator='\t'):
        self._dict_path = dict_path

        self._dict = {}  # 存放词典数据的dict，格式 word: (total_freq, ( (pos1, freq1), (pos2, freq2), ...) )

        self._total_freq = 0  # 总词频
        self._total_word = 0  # 总词数
        self._total_freq_log = float('-Inf')  # log形式的词频

        self._separator = separator

        self._init_dict_from_txt()

    def _init_dict_from_txt(self):
        """
        从txt词典里读取词，
        通过add_word方法，添加到 _dict
        并计算 _total_freq_log
        """
        with open(self._dict_path, 'r', encoding='utf-8') as dict_file:
            for line in dict_file:
                strs = line.strip().split(self._separator)  # 按照separator split
                word = strs[0]
                for i in range((len(strs) - 1) // 2):  # 对每个 pos,freq 对儿
                    pos, freq = strs[2 * i + 1], int(strs[2 * i + 2])
                    self.add_word(word, pos, freq)  # 调用add_word方法，添加到词典 _dict

        # 计算total_freq_log
        self._total_freq_log = math.log(self._total_freq or self._total_freq + 1)

    def add_word(self, word, pos, freq):
        """
        根据word pos freq，添加到 _dict
        考虑一词多性问题，考虑重复添加问题
        将前缀词也添加到_dict，方便计算
        """
        if not self._dict.get(word, None):  # 如果当前词没有被添加过
            self._dict[word] = (freq, ((pos, freq),))  # 直接设置freq 和 pos_freq对儿
            self._total_word += 1
            self._total_freq += freq
        else:  # 如果当前词被添加过
            pos_freq_pairs = []
            word_freq = self._dict[word][0]  # 之前添加的当前词的词频
            for pair in self._dict[word][1]:  # 迭代之前添加的当前词的 (词性,词频) 对

                # 当前词性被添加过, 取消之前的词频设置（后面覆盖前面）
                # （比如："今天 t 20" 之前没添加过一次；然后再次添加一个 "今天 t 100"）
                # 否则，追加到pos_freq_pairs
                if pair[0] == pos:
                    self._total_freq -= pair[1]
                    word_freq -= pair[1]
                else:
                    pos_freq_pairs.append(pair)

            # 追加当前 (词性,词频)对儿
            pos_freq_pairs.append((pos, freq))

            # 更新总词频，更新当前词总词频
            self._total_freq += freq
            word_freq += freq

            # 对pos_freq_pairs按照词频排序，更新到_dict
            self._dict[word] = (word_freq, tuple(sorted(pos_freq_pairs, key=lambda x: x[1], reverse=True)))

        # 将当前词的前缀词都添加到_dict，设置value为None
        # 便于词图构建时的判断
        # 比如 "台湾同乡会"，前缀为"台" "台湾" "台湾同" "台湾同乡"
        for i in range(len(word) - 1):
            pre_fix = word[:i + 1]
            if pre_fix not in self._dict:
                self._dict[pre_fix] = None

    def delete_word(self, word):
        """
        删除词
        """
        if word in self._dict:
            self._dict.pop(word)

    def freq(self, word):
        """
        获取当前词的词频
        """
        return self._dict[word][0] if word in self._dict and self._dict[word] else None

    def pos(self, word):
        """
        获取当前词的词性
        返回( (pos1, freq1), (pos2, freq2), ...)
        """
        return self._dict[word][1] if word in self._dict and self._dict[word] else None

    def first_pos_tag(self, word):
        """
        获取当前词的第一个词性
        """
        return self._dict[word][1][0][0] if word in self._dict and self._dict[word] else 'x'

    def is_in(self, word):
        """
        判断一个词是否在词典
        """
        return word in self._dict

    def get_total_freq_log(self):
        return self._total_freq_log

    def __str__(self):
        return json.dumps(
            {
                'dict': self._dict,
                'total_freq': self._total_freq,
                'total_word': self._total_word,
                'total_freq_log': self._total_freq_log
            },
            ensure_ascii=False)


if __name__ == '__main__':
    word_dict = WordDict('core_dict.txt')
    print(word_dict.freq('今天'))
    print(word_dict.is_in('老狼'))
    print(word_dict.pos('今天'))
