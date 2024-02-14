import re

from segment.word_tokenizer.word_based_tokenizer import WordBasedTokenizer


class WordSegEnglishTokenizer(WordBasedTokenizer):
    SENTENCE_PATTERN = '[\\.\\?!;]+\\s|\n+'
    WORD_PATTERN = '[,\\[\\]\\(\\)\\s]+'

    def __init__(self, complex_dict):
        super(WordSegEnglishTokenizer, self).__init__(complex_dict)

    def seg(self, sentence, enable_offset=False):
        return self._word_tokenize(sentence, enable_offset)

    @classmethod
    def _sentence_tokenize(cls, text, enable_offset):
        test_offset_list = []
        init_offset = 0
        start_offset = init_offset
        test_list = re.split('(' + cls.SENTENCE_PATTERN + ')', text)
        if enable_offset:
            for i in test_list:
                test_offset_list.append([i, start_offset, start_offset + len(i)])
                start_offset += len(i)
            assert start_offset == len(text), "offset error!"
            return test_offset_list
        else:
            return test_list

    @classmethod
    def _word_tokenize(cls, text, enable_offset):
        w = '(' + cls.WORD_PATTERN + '|' + cls.SENTENCE_PATTERN + ')'
        test_list = re.split(w, text)
        test_offset_list = []
        init_offset = 0
        start_offset = init_offset
        if enable_offset:
            for i in test_list:
                test_offset_list.append((i, start_offset, start_offset + len(i)))
                start_offset += len(i)
            assert start_offset == len(text), "offset error!"
            return test_offset_list
        else:
            return test_list
