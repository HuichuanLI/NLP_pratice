from abc import ABC

from segment.sequence_tokenizer.sequence_labeling_tokenizer import SequenceLabelingTokenizer


class SequenceSegTokenizer(SequenceLabelingTokenizer, ABC):

    def seg(self, content):
        """分词

        Args:
            content: str.输入文本

        """
        return self._tag(content)
