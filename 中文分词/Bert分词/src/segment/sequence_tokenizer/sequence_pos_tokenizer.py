from abc import ABC

from segment.sequence_tokenizer.sequence_labeling_tokenizer import SequenceLabelingTokenizer


class SequencePosTokenizer(SequenceLabelingTokenizer, ABC):

    def pos(self, content):
        """词性标注

        Args:
            content: str.输入文本

        """
        return self._tag(content)
