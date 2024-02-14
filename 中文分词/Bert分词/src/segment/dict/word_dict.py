import json
import math
import pickle

from segment import logger
from segment.dict.base_dict import BaseDict


class WordDict(BaseDict):
    """标准词典"""

    def __init__(self, dict_path=None, is_bin=False, separator='\t'):
        super(WordDict, self).__init__()
        self._dict_path = dict_path
        self._dict = {'#S#': (1, (('x', 1),)), '#E#': (1, (('x', 1),))}
        self._total_freq = 0
        self._total_word = 0
        self._total_freq_log = float('-Inf')
        self._is_bin = is_bin
        self._separator = separator
        if self._dict_path:
            self._init_dict()

    def _init_dict_from_txt(self):
        with open(self._dict_path, 'r', encoding='utf-8') as dict_file:
            for line in dict_file:
                try:
                    strs = line.strip().split(self._separator)
                    word = strs[0]
                    for i in range((len(strs) - 1) // 2):
                        pos, freq = strs[2 * i + 1], int(strs[2 * i + 2])
                        self.add_word(word, pos, freq)
                except Exception as e:
                    logger.exception('load word dict error for line: {}'.format(line))
        self._total_freq_log = math.log(self._total_freq or self._total_freq + 1)

    def _init_dict_from_pickle(self):
        with open(self._dict_path, 'rb') as dict_file:
            self._dict = pickle.load(dict_file)
        for k, v in self._dict.items():
            if v:
                self._total_freq += v[0]
        self._total_freq_log = math.log(self._total_freq or self._total_freq + 1)

    def _init_dict(self):
        if not self._is_bin:
            self._init_dict_from_txt()
        else:
            self._init_dict_from_pickle()

    def add_word(self, word, pos, freq):
        if not self._dict.get(word, None):
            self._dict[word] = (freq, ((pos, freq),))
            self._total_word += 1
            self._total_freq += freq
        else:
            pos_freq_pairs = []
            word_freq = self._dict[word][0]
            for pair in self._dict[word][1]:
                if pair[0] == pos:
                    self._total_freq -= pair[1]
                    word_freq -= pair[1]
                else:
                    pos_freq_pairs.append(pair)
            pos_freq_pairs.append((pos, freq))
            self._total_freq += freq
            word_freq += freq
            self._dict[word] = (word_freq, tuple(sorted(pos_freq_pairs, key=lambda x: x[1], reverse=True)))
        for i in range(len(word) - 1):
            pre_fix = word[:i + 1]
            if pre_fix not in self._dict:
                self._dict[pre_fix] = None

    def delete_word(self, word):
        if word in self._dict:
            self._dict.pop(word)

    def freq(self, word):
        return self._dict[word][0] if word in self._dict and self._dict[word] else None

    def pos(self, word):
        return self._dict[word][1] if word in self._dict and self._dict[word] else None

    def first_pos_tag(self, word):
        return self._dict[word][1][0][0] if word in self._dict and self._dict[word] else 'x'

    def is_in(self, word):
        return word in self._dict

    def get_total_freq_log(self):
        return self._total_freq_log

    def dump_txt_dict(self, output_path):
        with open(output_path, 'w', encoding='utf-8') as output_file:
            for word, info in self._dict.items():
                if word in {'#S#', '#E#'}:
                    continue
                if info:
                    output_file.write(word + self._separator + self._separator.join(
                        ['{}{}{}'.format(pos, self._separator, freq) for pos, freq in info[1]]) + '\n')

    def dump_bin_dict(self, output_path):
        with open(output_path, 'wb') as output_file:
            pickle.dump(self._dict, output_file)

    def __str__(self):
        return json.dumps(
            {
                'dict': self._dict,
                'total_freq': self._total_freq,
                'total_word': self._total_word,
                'total_freq_log': self._total_freq_log
            },
            ensure_ascii=False)

    @staticmethod
    def load(json_word_dict):
        wd = WordDict()
        word_dict_data = json.loads(json_word_dict)
        wd._dict = word_dict_data.get('dict', {'#S#': (1, (('x', 1),)), '#E#': (1, (('x', 1),))})
        wd._total_freq_log = word_dict_data.get('total_freq_log', float('-Inf'))
        wd._total_freq = word_dict_data.get('total_freq', 0)
        wd._total_word = word_dict_data.get('total_word', 0)
        return wd
