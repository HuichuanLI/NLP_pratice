from segment.word_tokenizer.word_seg_tokenizer import WordSegTokenizer


class WordSegSearchTokenizer(WordSegTokenizer):

    def __init__(self, complex_dict):
        super(WordSegSearchTokenizer, self).__init__(complex_dict)

    def _seg_big_word(self, content, regex_level):
        start_offset = 0
        for word in self._seg(content, seg_all=True, regex_level=regex_level):
            yield word, start_offset
            if len(word) > 2:
                for sub_word, sub_start_offset in self._seg_big_word(word, regex_level + 1):
                    yield sub_word, sub_start_offset + start_offset
            start_offset += len(word)

    def seg_all(self, word_pairs, start_offset, seg_all=True, enable_offset=False):
        for word_pair in word_pairs:
            yield word_pair
            if seg_all and len(word_pair[0]) > 2:
                for sub_word, sub_start_offset in self._seg_big_word(word_pair[0], 1):
                    new_sub_word = self._post_process(sub_word, enable_offset=enable_offset,
                                                      start_offset=start_offset + sub_start_offset)
                    if new_sub_word:
                        yield new_sub_word
            start_offset += len(word_pair[0])
