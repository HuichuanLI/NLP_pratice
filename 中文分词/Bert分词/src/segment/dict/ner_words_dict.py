import re

from segment.dict.word_dict import WordDict


class NERWordsDict(WordDict):

    def __init__(self):
        super(NERWordsDict, self).__init__()
        self._ner_seg_tags = 'S'
        self._length = 1
        self.regex_match_ner_seg = re.compile('^BM*E?$')
        self._ner_to_pos = {
            'PER': 'nr',
            'LOC': 'ns',
            'ORG': 'nt',
            'NZ': 'nz',
            'TIME': 't'
        }

    def load_ner_words(self, ner_words):
        self._ner_seg_tags = ['S']
        if ner_words:
            self._ner_seg_tags = ['S'] * (ner_words[-1][2] + 1)
            self._length = ner_words[-1][2] + 1
            for entity in ner_words:
                word, pos = entity[0], self._ner_to_pos.get(entity[3], entity[3])
                freq = 1
                self.add_word(word, pos, freq)
                beg, end = entity[1], entity[2]
                if len(word) > 1:
                    tmp_tags = 'B' + 'M' * (len(word) - 2) + 'E'
                    for i in range(beg, end):
                        self._ner_seg_tags[i + 1] = tmp_tags[i - beg]

        self._ner_seg_tags = ''.join(self._ner_seg_tags)

    def clear(self):
        self._dict = {}

    def is_ner_seg(self, begin_idx, end_idx):
        return self.regex_match_ner_seg.match(
            self._ner_seg_tags[begin_idx:end_idx]) if end_idx <= self._length else False
