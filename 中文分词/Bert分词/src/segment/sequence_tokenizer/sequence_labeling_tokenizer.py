import abc


class SequenceLabelingTokenizer(object, metaclass=abc.ABCMeta):
    """序列标注分词器基类"""

    @abc.abstractmethod
    def __init__(self):
        self._tagger = None

    @abc.abstractmethod
    def _tag(self, content):
        """标注

        Args:
            content: str.输入文本

        """
        pass
