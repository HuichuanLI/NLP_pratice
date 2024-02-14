from segment.model.crf.crfpp_predictor import CRFPPPredictor
from segment.sequence_tokenizer.sequence_seg_tokenizer import SequenceSegTokenizer
from segment.sequence_result_parser import SequenceResultParser


class CRFSegTokenizer(SequenceSegTokenizer):
    """docstring for CRFCutTokenizer."""

    def __init__(self, model_path):
        super(CRFSegTokenizer, self).__init__()
        self._predictor = CRFPPPredictor(model_path)

    def _tag(self, content):
        labels = self._predictor.predict([content])[0]
        return SequenceResultParser.parse(content, labels)
