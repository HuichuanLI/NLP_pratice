from segment.word_tokenizer.word_based_tokenizer import WordBasedTokenizer


class WordSegTokenizer(WordBasedTokenizer):

    def __init__(self, complex_dict):
        super(WordSegTokenizer, self).__init__(complex_dict)

    def seg(self, sentence, start_offset, enable_offset=False, model_seg=None, seg_all=False, regex_level=0,
            use_ner=False):
        """分词主函数

        Args:
            sentence:
            start_offset:
            enable_offset:
            model_seg: callable.模型切词方法
            seg_all:
            regex_level: int.正则级别
            use_ner:

        Returns:
            generator

        """
        for word in self._seg(sentence, model_seg=model_seg, seg_all=seg_all, regex_level=regex_level, use_ner=use_ner):
            yield self._post_process(word, enable_offset, start_offset)
            start_offset += len(word)

    @staticmethod
    def _post_process(word, enable_offset, start_offset):
        if enable_offset:
            return word, start_offset, start_offset + len(word)
        return word,
