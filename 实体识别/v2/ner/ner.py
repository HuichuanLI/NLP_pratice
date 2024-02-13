from itertools import chain

from ner import NER_TYPES, MODEL_TYPES
from ner import data
from ner import logger
from ner.model.crf.crfpp_predictor import CRFPPPredictor
from ner.ner_dict import NerDict
from ner.preprocess.text_splitter import TextSplitter
from ner.sequence_result_parser import SequenceResultParser
import re


class NER(object):

    def __init__(self, user_dict_path=None, dict_splitter='\t'):
        self._dict = NerDict(user_dict_path, dict_splitter)
        self._models = {}  # 装载多种模型，暂时只有CRF
        self._text_splitter = TextSplitter()
        self._crf_model_path = data.CRF_MODEL_PATH

    def _create_model(self, model):
        if model == 'CRF':
            self._models[model] = CRFPPPredictor(self._crf_model_path)
            logger.debug('CRF_MODEL local path: {}'.format(self._crf_model_path))

    @staticmethod
    def _check_input(**kwargs):
        if 'content' in kwargs:
            content, max_len = kwargs['content'], kwargs['max_len']
            assert len(content) <= max_len, '请确保输入content长度小于{}！'.format(max_len)
        if 'types' in kwargs:
            types = kwargs['types']
            assert isinstance(types, list) or isinstance(types, tuple), '请确保参数types是一个list或tuple！'
            for ner_type in types:
                assert ner_type in NER_TYPES, '请确保参数types的每个元素在{}中！'.format(NER_TYPES)
        if 'model' in kwargs:
            model = kwargs['model']
            assert model in MODEL_TYPES, '请确保参数model在{}中！'.format(MODEL_TYPES)
        if 'ner_type' in kwargs:
            ner_type = kwargs['ner_type']
            assert ner_type in NER_TYPES, '请确保参数ner_type在{}中！'.format(NER_TYPES)

    def find(self, content, types=NER_TYPES, model='CRF', enable_offset=False, max_len=int(1e4)):
        model = model.upper()
        self._check_input(types=types, model=model, content=content, max_len=max_len)

        # 模型结果
        model_result = self._find_by_model(content, types, model, enable_offset=True)
        # 字典结果
        dict_result = self._dict.findall(content, types, with_offset=True)
        # 合并
        result = self._merge_model_dict_result(model_result, dict_result, with_offset=enable_offset)
        # 规则干预
        self._rule_control(content, result, enable_offset)
        # 如果不返回索引，去重结果
        result = self._remove_duplicate(result, enable_offset)

        return result

    def find_by_model(self, content, types=NER_TYPES, model='CRF', enable_offset=False, max_len=int(1e4)):
        model = model.upper()
        self._check_input(types=types, model=model, content=content, max_len=max_len)

        result = self._find_by_model(content, types, model, enable_offset)

        result = self._remove_duplicate(result, enable_offset)

        return result

    def _find_by_model(self, content, types, model, enable_offset):
        result = {}

        sentences = self._text_splitter.split_sentence_merge_by_len(content, max_len=512)
        if model not in self._models:
            self._create_model(model)

        if model == 'CRF':
            labels = self._models[model].predict(sentences)
        else:
            raise ValueError

        terms = SequenceResultParser.parse_pos(list(chain(*sentences)), list(chain(*labels)), with_offset=True)

        for index, word, ner_type in terms:
            if ner_type not in types:
                continue
            if ner_type not in result:
                result[ner_type] = []
            result[ner_type].append([index, word] if enable_offset else word)

        return result

    def find_by_dict(self, content, types=NER_TYPES, enable_offset=False, max_len=int(1e4)):
        self._check_input(types=types, content=content, max_len=max_len)
        result = self._dict.findall(content, types, enable_offset)
        result = self._remove_duplicate(result, enable_offset)
        return result

    def add_word(self, word, ner_type):
        self._check_input(ner_type=ner_type)
        self._dict.add_word(word, ner_type)

    def delete_word(self, word, ner_type):
        self._check_input(ner_type=ner_type)
        self._dict.delete_word(word, ner_type)

    @staticmethod
    def _remove_duplicate(result, enable_offset):
        # 当结果不带索引时，去重
        if not enable_offset:
            for ner_type in result:
                result[ner_type] = list(set(result[ner_type]))
        return result

    @staticmethod
    def _merge_model_dict_result(model_result, dict_result, with_offset=True):
        """默认合并策略：将词典结果和模型结果合并，遇到冲突优先选词典的结果"""
        results = {}

        for ner_type in NER_TYPES:
            # 先将词典结果合并进results
            results[ner_type] = []
            results[ner_type].extend([term if with_offset else term[1] for term in dict_result.get(ner_type, [])])

            # 通过构建一个临时set，先存放所有的词典结果的下标
            tmp_set = set()
            for start, dict_word in dict_result.get(ner_type, []):
                tmp_set |= set(range(start, start + len(dict_word)))

            # 依次迭代模型结果，发现其下标已经在临时set里面了，
            for start, model_word in model_result.get(ner_type, []):
                if tmp_set & set(range(start, start + len(model_word))):  # 集合有交集，说明当前模型结果与词典结果索引有重叠，跳过
                    continue
                results[ner_type].append((start, model_word) if with_offset else model_word)

            # 假设结果为空，删除空列表
            if not results[ner_type]:
                del results[ner_type]

        return results

    def _rule_control(self, content, result, enable_offset):
        """
        规则控制，可以借助一些正则
        可以控制实体的前文、中间、后文，是什么 不是什么，来精准抽取实体

        特定领域的实体识别，比如合同甲方的提取，使用正则断言更为有用
        如：(?<=甲方为)(\\w{2,10})(?=，)
        """
        regex_list = {
            'PER': [
                '(?<=、)(栗战书)(?=、)',  # 前文必须是顿号、后文必须是顿号、中间是栗战书，抽取此实体为栗战书（此例子有点牵强）
            ]
        }
        for entity_type, rules in regex_list.items():
            for rule in rules:
                for match in re.finditer(rule, content):
                    result.get(entity_type, []).append(
                        (match.start(1), match.group(1)) if enable_offset else match.group(1)  # 选择group1为我们想要的
                    )
