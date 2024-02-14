import json
import math

import torch

from ner.model.base.base_predictor import BasePredictor
from ner.model.bert_crf.bert_crf_model import BertCRFModel
from ner.model.vocab import Vocab


class BERTCRFPredictor(BasePredictor):
    def __init__(self, pretrained_model_dir, model_dir, vocab_name='vocab.json'):
        self.pretrained_model_dir = pretrained_model_dir
        self.model_dir = model_dir

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.vocab = Vocab()

        self._load_config()
        self.vocab.load_vocab('{}/{}'.format(model_dir, vocab_name))
        self._load_model()

    def _load_config(self):
        with open('{}/train_config.json'.format(self.model_dir), 'r') as f:
            self._config = json.loads(f.read())

    def _load_model(self):
        self.model = BertCRFModel(self.pretrained_model_dir, self._config['label_size'])
        self.model.load_state_dict(
            torch.load('{}/{}'.format(self.model_dir, self._config['ckpt_name']), map_location=self.device)
        )
        self.model.eval()

        self.model.to(self.device)

        self.vocab.set_unk_vocab_id(self.vocab.vocab2id['[UNK]'])
        self.vocab.set_pad_vocab_id(self.vocab.vocab2id['[PAD]'])

    def predict(self, texts, batch_size=64, max_len=512):
        """

        Args:
            texts: list[list[str]].预测样本，如果输入被切分，请确保是使用bert的tokenizer切分的
            batch_size: int.
            max_len: int.最大序列长度（请和bert预训练模型中的max_position_embeddings保持一致）

        Returns:
            list[list[str]].标签序列

        """
        batch_labels = []

        for batch_idx in range(math.ceil(len(texts) / batch_size)):
            text_batch = texts[batch_size * batch_idx: batch_size * (batch_idx + 1)]

            # 当前batch最大长度
            batch_max_len = min(max([len(text) for text in text_batch]) + 2, max_len)

            batch_input_ids, batch_att_mask = [], []
            for text in text_batch:
                assert isinstance(text, list)
                text = ' '.join(text)  # 确保输入encode_plus函数为文本

                encoded_dict = self.model.bert_tokenizer.encode_plus(text, max_length=batch_max_len,
                                                                     pad_to_max_length=True,
                                                                     return_tensors='pt', truncation=True)
                batch_input_ids.append(encoded_dict['input_ids'])
                batch_att_mask.append(encoded_dict['attention_mask'])
            batch_input_ids = torch.cat(batch_input_ids)
            batch_att_mask = torch.cat(batch_att_mask)

            batch_input_ids, batch_att_mask = batch_input_ids.to(self.device), batch_att_mask.to(self.device)

            with torch.no_grad():
                logits = self.model(batch_input_ids, batch_att_mask)
                for logit, att_mask in zip(logits, batch_att_mask):
                    active_logit = logit[att_mask == 1][1:-1]  # 去除pad部分、[CLS]和[SEP]部分
                    labels = [self.vocab.id2tag[label_id.item()] for label_id in torch.argmax(active_logit, dim=1)]
                    batch_labels.append(labels)

        return batch_labels

    def get_bert_tokenizer(self):
        return self.model.bert_tokenizer
