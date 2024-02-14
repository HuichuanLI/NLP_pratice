from segment import logger as _logger
from segment.preprocess.text_splitter import TextSplitter
from segment import data
from segment.dict.complex_dict import ComplexDict
from segment.dict.model_words_dict import ModelWordsDict
from segment.dict.ner_words_dict import NERWordsDict
from segment.dict.regex_dict import RegexDict
from segment.dict.stop_words import StopWords
from segment.dict.word_dict import WordDict
from segment.sequence_tokenizer.crf_pos_tokenizer import CRFPosTokenizer
from segment.sequence_tokenizer.crf_seg_tokenizer import CRFSegTokenizer
from segment.sequence_tokenizer.dl_pos_tokenizer import DLPosTokenizer
from segment.sequence_tokenizer.dl_seg_tokenizer import DLSegTokenizer
from segment.sequence_tokenizer.hmm_pos_tokenizer import HMMPosTokenizer
from segment.sequence_tokenizer.hmm_seg_tokenizer import HMMSegTokenizer
from segment.word_tokenizer.word_pos_tokenizer import WordPosTokenizer
from segment.word_tokenizer.word_seg_english_tokenizer import WordSegEnglishTokenizer
from segment.word_tokenizer.word_seg_search_tokenizer import WordSegSearchTokenizer
from segment.word_tokenizer.word_seg_tokenizer import WordSegTokenizer


class Segment(object):

    def __init__(self, user_dict_path=None, core_dict=None, regex_dict=None, new_words_dict=None,
                 stop_words=None, logger=_logger):
        self.logger = logger
        self.logger.info('start segment initialize')
        self._init_dict(user_dict_path, core_dict, regex_dict, new_words_dict, stop_words)
        self._text_splitter = TextSplitter(stops='([，,。？?！!；;：:\n ])')
        self._stop_signs = set(self._text_splitter.stops[2:-2])
        self._init_tokenizer()
        self.logger.info('finish segment initialize with user dict path: {}'.format(str(user_dict_path)))

    def _init_dict(self, user_dict_path, core_dict, regex_dict, new_words_dict, stop_words):
        self._user_dict = WordDict(user_dict_path)
        self._core_dict = core_dict or WordDict(data.CORE_DICT_PATH)
        self._regex_dict = regex_dict or RegexDict(data.REGEX_DICT_PATH)
        self._new_words_dict = new_words_dict or WordDict()  # TODO WordDict(data.NEW_WORDS_DICT_PATH)
        self._model_word_dict = ModelWordsDict()
        self._ner_word_dict = NERWordsDict()
        self._complex_dict = ComplexDict(self._core_dict, self._user_dict, self._regex_dict, self._new_words_dict,
                                         self._model_word_dict, self._ner_word_dict)
        self._stop_words = stop_words or StopWords(data.STOP_WORDS_PATH)

    def _init_tokenizer(self):
        self._seg_tokenizer = WordSegTokenizer(self._complex_dict)
        self._pos_tokenizer = WordPosTokenizer(self._complex_dict)
        self._seg_english_tokenizer = WordSegEnglishTokenizer(self._complex_dict)
        self._seg_search_tokenizer = WordSegSearchTokenizer(self._complex_dict)
        self._crf_seg_tokenizer = None
        self._crf_pos_tokenizer = None
        self._hmm_seg_tokenizer = None
        self._hmm_pos_tokenizer = None
        self._dl_seg_tokenizer = None
        self._dl_pos_tokenizer = None

    def _create_model_seg(self, model):
        model_seg = None
        if model == 'CRF':
            if not self._crf_seg_tokenizer:
                self._crf_seg_tokenizer = CRFSegTokenizer(data.CRF_CUT_MODEL_PATH)
            model_seg = self._crf_seg_tokenizer.seg
        elif model == 'HMM':
            if not self._hmm_seg_tokenizer:
                self._hmm_seg_tokenizer = HMMSegTokenizer(data.HMM_CUT_MODEL_PATH)
            model_seg = self._hmm_seg_tokenizer.seg
        elif model == 'DL':
            if not self._dl_seg_tokenizer:
                self._dl_seg_tokenizer = DLSegTokenizer(data.DL_CUT_MODEL_PATH)
            model_seg = self._dl_seg_tokenizer.seg
        return model_seg

    def _create_model_pos(self, model):
        model_pos = None
        if model == 'CRF':
            if not self._crf_pos_tokenizer:
                self._crf_pos_tokenizer = CRFPosTokenizer(data.CRF_POS_MODEL_PATH)
            model_pos = self._crf_pos_tokenizer.pos
        elif model == 'HMM':
            if not self._hmm_pos_tokenizer:
                self._hmm_pos_tokenizer = HMMPosTokenizer(data.HMM_POS_MODEL_PATH)
            model_pos = self._hmm_pos_tokenizer.pos
        elif model == 'DL':
            if not self._dl_pos_tokenizer:
                self._dl_pos_tokenizer = DLPosTokenizer(data.DL_POS_MODEL_PATH)
            model_pos = self._dl_pos_tokenizer.pos
        return model_pos

    def _seg(self, sentence, model=None, enable_offset=False, start_offset=0, enable_stop_word=False, seg_all=False,
             regex_level=0, use_ner=False):
        model_seg = self._create_model_seg(model)
        if sentence in self._stop_signs:
            word_pairs = [(sentence, start_offset, start_offset + 1)] if enable_offset else [(sentence,)]
        else:
            word_pairs = self._seg_tokenizer.seg(sentence, start_offset, enable_offset=enable_offset,
                                                 model_seg=model_seg, seg_all=seg_all, regex_level=regex_level,
                                                 use_ner=use_ner)
        if enable_stop_word:
            return self._filter_stop_words(word_pairs)
        return word_pairs

    def _seg_for_search(self, sentence, init_offset=0, seg_all=True, model=None, enable_stop_word=False,
                        enable_offset=True, use_ner=False):
        start_offset = init_offset
        word_pairs = self._seg(sentence, start_offset=start_offset, model=model, enable_offset=enable_offset,
                               use_ner=use_ner)
        all_word_pairs = self._seg_search_tokenizer.seg_all(
            word_pairs, start_offset, seg_all=seg_all, enable_offset=enable_offset)
        if enable_stop_word:
            return self._filter_stop_words(all_word_pairs)
        return all_word_pairs

    def _pos(self, sentence, seg_model=None, enable_offset=False, start_offset=0, enable_stop_word=False,
             enhance=True, use_ner=False):
        model_pos = self._create_model_pos(seg_model)
        if sentence in self._stop_signs:
            words = [(sentence, start_offset, start_offset + 1, 'w')] if enable_offset else [(sentence, 'w')]
        else:
            words = self._pos_tokenizer.pos(sentence, start_offset, enable_offset=enable_offset, model_seg=model_pos,
                                            enhance=enhance, use_ner=use_ner)
        if enable_stop_word:
            return self._filter_stop_words(words)
        return words

    def seg(self, content, model=None, enable_offset=False, enable_stop_word=False, use_ner=False):
        """分词主入口

        Args:
            content: str.输入文本
            model: str.模型类别，暂时支持HMM, CRF, DL, BERT（大小写都可以）
            enable_offset: bool.是否启用索引模式
            enable_stop_word: bool.是否过滤停用词
            use_ner: bool.是否使用ner辅助分词

        Returns:
            generator

        """
        if model:
            model = model.upper()
        init_offset = 0
        for sentence in self._text_splitter.split_sentence_for_seg(content):
            for word in self._seg(sentence, model, enable_offset=enable_offset, start_offset=init_offset,
                                  enable_stop_word=enable_stop_word, use_ner=use_ner):
                if not enable_offset:
                    yield word[0]
                else:
                    yield word
            init_offset += len(sentence)

    def seg_for_search(self, content, model=None, enable_offset=True, enable_stop_word=False, seg_all=True,
                       use_ner=False):
        """搜索场景下的分词，即多粒度分词

        Args:
            content: str.输入文本
            model: str.模型类别，暂时支持HMM, CRF, DL, BERT（大小写都可以）
            enable_offset: bool.是否启用索引模式
            enable_stop_word: bool.是否过滤停用词
            seg_all: bool.是否启用多粒度模式
            use_ner: bool.是否使用ner辅助分词

        Returns:
            generator

        """
        if model:
            model = model.upper()
        init_offset = 0
        for sentence in self._text_splitter.split_sentence_for_seg(content):
            for word in self._seg_for_search(sentence, init_offset, seg_all=seg_all, model=model,
                                             enable_stop_word=enable_stop_word, enable_offset=enable_offset,
                                             use_ner=use_ner):
                if not enable_offset:
                    yield word[0]
                else:
                    yield word
            init_offset += len(sentence)

    def seg_for_english(self, content, enable_offset=False, enable_stop_word=False):
        """英文场景下的分词，通过正则实现

        Args:
            content: str.输入文本
            enable_offset: bool.是否启用索引模式
            enable_stop_word: bool.是否过滤停用词

        Returns:
            generator

        """
        word_pairs = self._seg_english_tokenizer.seg(content, enable_offset=enable_offset)
        if enable_stop_word:
            return self._filter_stop_words(word_pairs)
        return word_pairs

    def pos(self, content, model=None, enable_offset=False, enable_stop_word=False, enhance=True, use_ner=False):
        """带词性的分词

        Args:
            content: str.输入文本
            model: str.模型类别，暂时支持HMM, CRF（大小写都可以）
            enable_offset: bool.是否启用索引模式
            enable_stop_word: bool.是否过滤停用词
            enhance: bool.是否启用enhance模式（enhance模式解决一词多性问题，如果启用，使用hmm模型，否则使用高频词性）
            use_ner: bool.是否使用ner辅助分词

        Returns:
            generator

        """
        if model:
            model = model.upper()
        init_offset = 0
        for sentence in self._text_splitter.split_sentence_for_seg(content):
            for word in self._pos(sentence, seg_model=model, enable_offset=enable_offset, start_offset=init_offset,
                                  enable_stop_word=enable_stop_word, enhance=enhance, use_ner=use_ner):
                yield word
            init_offset += len(sentence)

    def add_word(self, word, pos, freq):
        """添加词到词典

        Args:
            word: str.词
            pos: str.词性
            freq: int.词频

        """
        self._user_dict.add_word(word, pos, freq)

    def disable_word(self, word):
        """禁用某个词

        Args:
            word: str.词

        """
        self._complex_dict.disable_word(word)

    def restore_word(self, word):
        """撤销禁用词

        Args:
            word: str.词

        """
        self._complex_dict.restore_word(word)

    def _filter_stop_words(self, word_pairs):
        for word_pair in word_pairs:
            if not self._stop_words.is_in(word_pair[0]):
                yield word_pair
