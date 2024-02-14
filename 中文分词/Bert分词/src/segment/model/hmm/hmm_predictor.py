from segment.viterbi import Viterbi
from segment.model.base.base_predictor import BasePredictor
from segment.model.hmm.hmm_model import HMMModel


class HMMPredictor(BasePredictor):

    def __init__(self,
                 model_path,
                 prev_tags=(('B', 'ES'), ('M', 'MB'), ('S', 'SE'), ('E', 'BM')),
                 start_tags='BS',
                 end_tags='ES'):
        self._model_path = model_path
        self._prev_tags = dict(prev_tags)
        self._end_tags = end_tags
        self._model = HMMModel(self._model_path)
        self._viterbi = Viterbi(self._model.get_tags(), self._model.get_start_p(), self._model.get_trans_p(),
                                self._prev_tags, start_tags, end_tags)

    def _gen_graph_from_sentence(self, sentence):
        sentence_len = len(sentence)
        score_nodes = [self._model.get_emit(sentence[idx]) for idx in range(sentence_len)]
        return score_nodes

    def predict(self, texts):
        """

        Args:
            texts: list[list[str]].预测样本

        Returns:
            list[list[str]].标签序列

        """
        results = []

        for text in texts:
            if not text:
                result = []
            else:
                score_nodes = self._gen_graph_from_sentence(text)
                best_prob, result = self._viterbi.parse(score_nodes)
            results.append(result)

        return results
