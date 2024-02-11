import copy

from crfpp_model import CRFPPModel
from viterbi import Viterbi


class CRFPPPredictor(object):

    def __init__(self,
                 model_path,
                 prev_tags=(('B', 'ES'), ('M', 'MB'), ('S', 'SE'), ('E', 'BM')),  # 转移限制（尽管转移矩阵已经隐含这些信息，这里可以确保不出现异常标签转移）
                 start_tags='BS',  # 可能的起始标签
                 end_tags='ES'):  # 可能的结束标签
        self._model_path = model_path

        # 加载模型
        self._model = CRFPPModel(self._model_path)

        # 初始化维特比算法
        self._viterbi = Viterbi(
            self._model.get_tags()[:],  # 传入可能的标签
            {},
            copy.deepcopy(self._model.get_trans_func_weight()),  # 传入转移矩阵
            dict(prev_tags), start_tags, end_tags  # 传入上面的标签限制
        )

    def _gen_graph(self, pairs):
        """
        根据文本，得到篱笆网络，生成发射矩阵
        """
        pairs_len = len(pairs)
        score_nodes = [
            # 对每个时刻计算发射score
            self._model.compute_score(self._model.gen_feature(pairs, pairs_len, idx)) for idx in range(pairs_len)
        ]
        return score_nodes

    def predict(self, texts):
        """
        批量计算crf解码

        生成篱笆网络中每个位置的发射score
        将转移矩阵传入维特比算法，进行解码
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


if __name__ == '__main__':
    predictor = CRFPPPredictor('model/crf-seg.model.txt')
    texts = ['今天天气不错', '你好么']
    results = predictor.predict(texts)
    print(results)
