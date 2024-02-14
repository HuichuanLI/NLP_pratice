from segment.model.crf.crfpp_predictor import CRFPPPredictor
from segment.sequence_tokenizer.sequence_pos_tokenizer import SequencePosTokenizer
from segment.sequence_result_parser import SequenceResultParser


class CRFPosTokenizer(SequencePosTokenizer):
    """docstring for CRFPosTokenizer."""

    def __init__(self, model_path):
        super(CRFPosTokenizer, self).__init__()
        self._predictor = CRFPPPredictor(model_path)

    def _tag(self, content):
        labels = self._predictor.predict([content])[0]
        return SequenceResultParser.parse_pos(content, labels)
