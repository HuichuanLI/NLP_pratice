import json
import math

import torch
from torch import LongTensor

from ner.model.base.base_predictor import BasePredictor
from ner.model.idcnn.idcnn_crf import IDCNN_CRF
from ner.model.vocab import Vocab


class IDCNNPredictor(BasePredictor):
    def __init__(self, model_dir, vocab_name='vocab.json'):
        self.model_dir = model_dir

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.vocab = Vocab()

        # 加载config，vocab，model
        self._load_config()
        self.vocab.load_vocab('{}/{}'.format(model_dir, vocab_name))
        self._load_model()

    def _load_config(self):
        """加载训练时的config"""
        with open('{}/train_config.json'.format(self.model_dir), 'r') as f:
            self._config = json.loads(f.read())

    def _load_model(self):
        """根据config和vocab，加载模型"""
        self.model = IDCNN_CRF(
            seq_len=self._config['max_len'], filters=self._config['filters'], hidden_num=self._config['hidden_num'],
            vocab_size=self._config['vocab_size'], embedding_size=self._config['embedding_size'],
            label_size=self._config['label_size'], dropout_rate=self._config['dropout_rate']
        )
        self.model.load_state_dict(
            torch.load('{}/{}'.format(self.model_dir, self._config['ckpt_name']), map_location=self.device)
        )
        # 设置为评估模式（预测模式）
        self.model.eval()
        # 拷贝模型到当前设备
        self.model.to(self.device)

    def predict(self, texts, batch_size=64):
        """

        Args:
            texts: list[list[str]].预测样本
            batch_size: int.

        Returns:
            list[list[str]].标签序列

        """
        batch_labels = []

        for batch_idx in range(math.ceil(len(texts) / batch_size)):  # 分batch
            text_batch = texts[batch_size * batch_idx: batch_size * (batch_idx + 1)]  # 当前batch的所有文本

            # 根据vocab，将原始文本转为id形式
            batch_input_ids, input_lens = [], []
            for text in text_batch:
                batch_input_ids.append([])
                input_lens.append(len(text))
                for seg in text:
                    batch_input_ids[-1].append(self.vocab.vocab2id.get(seg, self.vocab.unk_vocab_id))
                # pad到最大
                batch_input_ids[-1] += [self.vocab.pad_vocab_id] * (self._config['max_len'] - len(batch_input_ids[-1]))

            # 转为tensor
            batch_input_ids, input_lens = LongTensor(batch_input_ids), LongTensor(input_lens)
            # 将数据拷贝到指定设备
            batch_input_ids, input_lens = batch_input_ids.to(self.device), input_lens.to(self.device)

            with torch.no_grad():
                # 前向推理，得到预测的labels
                best_path = self.model(batch_input_ids, input_lens)
                for predict_labels, input_len in zip(best_path, input_lens):
                    # 将id形式的labels转为str形式，并截掉pad部分
                    batch_labels.append(
                        [self.vocab.id2tag[label_id] for label_id in predict_labels[:input_len].tolist()]
                    )

        return batch_labels
