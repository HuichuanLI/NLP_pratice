from ahocorasick import Automaton

from ner import NER_TYPES


class NerDict(object):
    """
    通过AC自动机，实现大批量词典的快速加载、检索
    """

    def __init__(self, dict_path=None, splitter='\t'):
        self._splitter = splitter
        self.autos = {ner_type: None for ner_type in NER_TYPES}  # 每个实体类型对应一个自动机
        self.words = {ner_type: set() for ner_type in NER_TYPES}  # 每个实体类型对应一个词集合
        if dict_path:
            self._load(dict_path)

    def _load(self, dict_path):
        # 按行读取词典，将实体词加载到words
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                word, ner_type = line.split(self._splitter)[:2]
                self.words[ner_type].add(word)
        # 根据加载好的words，构建自动机
        for ner_type in NER_TYPES:
            if not self.words[ner_type]:
                continue
            self.autos[ner_type] = self._build_auto(self.words[ner_type])

    @staticmethod
    def _build_auto(words):
        """根据词典，构建自动机"""
        auto = Automaton()
        [auto.add_word(word, word) for word in words]
        auto.make_automaton()
        return auto

    def findall(self, content, types=NER_TYPES, with_offset=True):
        """从文本中查找实体词"""
        results = {}
        for ner_type in types:  # 一个类型一个类型处理
            if not self.autos[ner_type]:
                continue
            # 迭代自动机的结果
            results[ner_type] = []
            for end_idx, word in self.autos[ner_type].iter(content):  # 自动机返回的是终止索引，并且包含终止位置
                results[ner_type].append((end_idx + 1 - len(word), word) if with_offset else word)
            # 此类型没找到结果，删除空列表
            if not results[ner_type]:
                del results[ner_type]
        return results

    def add_word(self, word, ner_type):
        """加词，重新构建自动机"""
        self.words[ner_type].add(word)
        self.autos[ner_type] = self._build_auto(self.words[ner_type])

    def delete_word(self, word, ner_type):
        """删词，重新构建自动机"""
        self.words[ner_type].remove(word)
        if self.words[ner_type]:
            self.autos[ner_type] = self._build_auto(self.words[ner_type])
        else:
            self.autos[ner_type] = None
