import copy

from ner.model.base.base_predictor import BasePredictor
from ner.model.crf.crfpp_model import CRFPPModel
from ner.viterbi import Viterbi


class CRFPPPredictor(BasePredictor):

    def __init__(self,
                 model_path,
                 prev_tags=(('B', 'ES'), ('M', 'MB'), ('S', 'SE'), ('E', 'BM')),
                 start_tags='BS',
                 end_tags='ES'):
        self._model_path = model_path
        self._model = CRFPPModel(self._model_path, is_json=True)
        self._viterbi = Viterbi(self._model.get_tags()[:], {}, copy.deepcopy(self._model.get_trans_func_weight()),
                                dict(prev_tags), start_tags, end_tags)

    def _gen_graph(self, pairs):
        pairs_len = len(pairs)
        score_nodes = [
            self._model.compute_score(self._model.gen_feature(pairs, pairs_len, idx)) for idx in range(pairs_len)
        ]
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
                score_nodes = self._gen_graph(text)
                best_prob, result = self._viterbi.parse(score_nodes)
            results.append(result)

        return results
