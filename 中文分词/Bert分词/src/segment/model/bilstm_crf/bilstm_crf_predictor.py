import json
import math

import torch

from segment.model.base.base_predictor import BasePredictor
from segment.model.bilstm_crf.bilstm_crf_model import BiLSTMCRFModel


class BiLSTMCRFPredictor(BasePredictor):
    def __init__(self, model_dir):
        self.model_dir = model_dir

        self._load_vocab()
        self._load_params()
        self._load_model()

    def _load_vocab(self):
        with open('{}/vocab.json'.format(self.model_dir), 'r', encoding='utf-8') as f:
            self._vocab = json.load(f)

    def _load_params(self):
        with open('{}/params.json'.format(self.model_dir), 'r', encoding='utf-8') as f:
            self._params = json.load(f)

    def _load_model(self):
        self.model = BiLSTMCRFModel(
            vocab_size=len(self._vocab['word2id']),
            tag_to_ix=self._vocab['tag2id'],
            embedding_dim=self._params['embedding_dim'],
            hidden_dim=self._params['hidden_num']
        )
        self.model.load_state_dict(torch.load('{}/model.bin'.format(self.model_dir)))

    def predict(self, texts, batch_size=64, max_len=512):
        paths = []

        for batch_idx in range(math.ceil(len(texts) / batch_size)):
            text_batch = texts[batch_size * batch_idx: batch_size * (batch_idx + 1)]
            for text in text_batch:
                text_tensor = torch.tensor(
                    [self._vocab['word2id'].get(w, self._vocab['word2id']['<UNK>']) for w in text], dtype=torch.long
                )
                score, tags = self.model(text_tensor)
                tags = [self._vocab['id2tag'][str(tag)] for tag in tags]
                paths.append(tags)

        return paths
