import math
import pickle
import re

from segment.dict.word_dict import WordDict


class RegexDict(WordDict):

    def __init__(self, dict_path=None, is_bin=False):
        self._regex_word_pairs = []
        super(RegexDict, self).__init__(dict_path, is_bin)
        self._dict = {}

    def _init_dict_from_txt(self):
        with open(self._dict_path, 'r', encoding='utf-8') as regex_dict_file:
            for line in regex_dict_file:
                line = line.strip()
                if not line or line[0] == '#':
                    continue
                strs = line.strip().split('  ')
                self._regex_word_pairs.append((int(strs[0]), re.compile(strs[1]), strs[2], strs[3], int(strs[4])))

    def _init_dict_from_pickle(self):
        with open(self._dict_path, 'rb') as dict_file:
            self._regex_word_pairs = pickle.load(dict_file)
        for k, v in self._dict.items():
            if v:
                self._total_freq += v[0]
        self._total_freq_log = math.log(self._total_freq)

    def dump_bin_dict(self, output_path):
        with open(output_path, 'wb') as output_file:
            pickle.dump(self._regex_word_pairs, output_file)

    def get_regex_word_pairs(self):
        return self._regex_word_pairs

    def clear(self):
        self._dict = {}
