from segment.model.hmm.hmm_predictor import HMMPredictor
from segment.sequence_tokenizer.sequence_pos_tokenizer import SequencePosTokenizer
from segment.sequence_result_parser import SequenceResultParser


class HMMPosTokenizer(SequencePosTokenizer):
    """docstring for HMMPosTokenizer."""

    def __init__(self, model_path):
        super(HMMPosTokenizer, self).__init__()
        self._predictor = HMMPredictor(model_path)

    def _tag(self, content):
        labels = self._predictor.predict([content])[0]
        return SequenceResultParser.parse_pos(content, labels)
