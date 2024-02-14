import re

from segment.dict.word_dict import WordDict


class ModelWordsDict(WordDict):

    def __init__(self):
        super(ModelWordsDict, self).__init__()
        self._model_seg_tags = 'S'
        self.regex_match_model_seg = re.compile('^BM*E?$')

    def load_model_words(self, model_words):
        self._model_seg_tags = 'S'
        for word in model_words:
            if isinstance(word, tuple):
                word, pos = word
            else:
                word, pos = word, None
            freq = 50
            self.add_word(word, pos, freq)
            if len(word) == 1:
                self._model_seg_tags += 'S'
            else:
                self._model_seg_tags += ('B' + 'M' * (len(word) - 2) + 'E')

    def clear(self):
        self._dict = {}

    def is_model_seg(self, begin_idx, end_idx):
        return self.regex_match_model_seg.match(self._model_seg_tags[begin_idx:end_idx])
