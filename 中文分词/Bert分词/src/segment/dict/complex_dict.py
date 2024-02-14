from segment.dict.base_dict import BaseDict


class ComplexDict(BaseDict):
    """复合词典"""

    def __init__(self, core_dict, user_dict, regex_dict, new_words_dict, model_word_dict, ner_word_dict):
        super(ComplexDict, self).__init__()
        self._core_dict = core_dict
        self._user_dict = user_dict
        self._regex_dict = regex_dict
        self._new_words_dict = new_words_dict
        self._model_word_dict = model_word_dict
        self._ner_word_dict = ner_word_dict
        self._disabled_word_list = []

    def freq(self, word):
        if word in self._disabled_word_list:
            return None
        return self._regex_dict.freq(word) or self._user_dict.freq(word) or self._core_dict.freq(word) \
            or self._new_words_dict.freq(word) or self._model_word_dict.freq(word) or self._ner_word_dict.freq(word)

    def pos(self, word):
        return self._regex_dict.pos(word) or self._user_dict.pos(word) or self._core_dict.pos(
            word) or self._model_word_dict.pos(word) or self._ner_word_dict.pos(word) or (('x', 1),)

    def is_in(self, word):
        if self._core_dict.is_in(word):
            return 'core_dict'
        elif self._user_dict.is_in(word):
            return 'user_dict'
        elif self._regex_dict.is_in(word):
            return 'regex_dict'
        elif self._new_words_dict.is_in(word):
            return 'new_words_dict'
        elif self._model_word_dict.is_in(word):
            return 'model_word_dict'
        elif self._ner_word_dict.is_in(word):
            return 'ner_word_dict'
        else:
            return None

    def get_total_freq_log(self):
        return self._core_dict.get_total_freq_log()

    def get_regex_dict(self):
        return self._regex_dict

    def get_model_dict(self):
        return self._model_word_dict

    def get_ner_dict(self):
        return self._ner_word_dict

    def disable_word(self, word):
        self._disabled_word_list.append(word)

    def restore_word(self, word):
        if word in self._disabled_word_list:
            self._disabled_word_list.remove(word)
