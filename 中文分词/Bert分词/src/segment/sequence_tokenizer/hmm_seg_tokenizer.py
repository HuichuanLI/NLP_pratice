from segment.model.hmm.hmm_predictor import HMMPredictor
from segment.sequence_tokenizer.sequence_seg_tokenizer import SequenceSegTokenizer
from segment.sequence_result_parser import SequenceResultParser


class HMMSegTokenizer(SequenceSegTokenizer):
    """docstring for HMMCutTokenizer."""

    def __init__(self, model_path):
        super(HMMSegTokenizer, self).__init__()
        self._predictor = HMMPredictor(model_path)

    def _tag(self, content):
        labels = self._predictor.predict([content])[0]
        return SequenceResultParser.parse(content, labels)
