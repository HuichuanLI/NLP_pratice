import json
from datetime import datetime

from segment import logger


class HMMModel(object):

    def __init__(self, model_path):
        self._model_path = model_path
        start_time = datetime.now()
        logger.debug('begin loading HMM model')
        self._load_model()
        end_time = datetime.now()
        logger.debug('end loading HMM model, cost time: {}'.format(end_time - start_time))

    def _load_model(self):
        self._start_p = json.load(open('{}/start_p.json'.format(self._model_path)))
        self._trans_p = json.load(open('{}/trans_p.json'.format(self._model_path)))
        self._emit_p = json.load(open('{}/emit_p.json'.format(self._model_path)))
        self._tags = tuple([k for k in self._emit_p])
        self._min_emit_p = min([min(v.values()) for k, v in self._emit_p.items()])
        self._min_emit_p *= 10

    def get_tags(self):
        return self._tags

    def get_start_p(self):
        return self._start_p

    def get_trans_p(self):
        return self._trans_p

    def get_emit(self, node):
        score = {}
        for idx, tag in enumerate(self._tags):
            score[tag] = self._emit_p[tag].get(node, self._min_emit_p)
        return score
