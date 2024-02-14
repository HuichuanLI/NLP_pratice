from segment.model.bilstm_crf.bilstm_crf_predictor import BiLSTMCRFPredictor
from segment.sequence_tokenizer.sequence_pos_tokenizer import SequencePosTokenizer
from segment.sequence_result_parser import SequenceResultParser


class DLPosTokenizer(SequencePosTokenizer):
    """docstring for CRFPosTokenizer."""

    def __init__(self, model_path):
        super(DLPosTokenizer, self).__init__()
        self._predictor = BiLSTMCRFPredictor(model_path)

    def _tag(self, content):
        labels = self._predictor.predict([content])[0]
        return SequenceResultParser.parse_pos(content, labels)
