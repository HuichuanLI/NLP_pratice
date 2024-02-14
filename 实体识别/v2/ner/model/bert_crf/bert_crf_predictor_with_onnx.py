import json
import math

import torch
import torch.onnx
from onnxruntime import InferenceSession
from torch.nn import DataParallel

from ner.model.base.base_predictor import BasePredictor
from ner.model.bert_crf.bert_crf_model import BertCRFModel
from ner.model.vocab import Vocab


class BERTCRFPredictor(BasePredictor):
    def __init__(self, pretrained_model_dir, model_dir, vocab_name='vocab.json',
                 enable_parallel=False):
        self.pretrained_model_dir = pretrained_model_dir
        self.model_dir = model_dir
        self.enable_parallel = enable_parallel

        self.device = torch.device('cpu')  # # 仅测试cpu下

        self.vocab = Vocab()

        self._load_config()
        self.vocab.load_vocab('{}/{}'.format(model_dir, vocab_name))
        self._load_model()

        self.is_cur_onnx_model = False  # # 当前是否是onnx模型
        self.onnx_model = None  # # onnx_model初始化为None

    def _load_config(self):
        with open('{}/train_config.json'.format(self.model_dir), 'r') as f:
            self._config = json.loads(f.read())

    def _load_model(self):
        self.model = BertCRFModel(
            self.pretrained_model_dir, self._config['label_size'],
            loss_type=self._config['loss_type'], focal_loss_alpha=self._config['focal_loss_alpha'],
            focal_loss_gamma=self._config['focal_loss_gamma']
        )
        self.model.load_state_dict(
            torch.load('{}/{}'.format(self.model_dir, self._config['ckpt_name']), map_location=self.device)
        )
        self.model.eval()

        if self.enable_parallel:
            self.model = DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))

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
            # batch_max_len = min(max([len(text) for text in text_batch]) + 2, max_len)
            batch_max_len = max_len

            batch_input_ids, batch_att_mask = [], []
            for text in text_batch:
                assert isinstance(text, list)
                text = ' '.join(text)  # 确保输入encode_plus函数为文本
                # 根据是否并行，获取bert_tokenizer
                bert_tokenizer = self.model.bert_tokenizer if not self.enable_parallel \
                    else self.model.module.bert_tokenizer
                encoded_dict = bert_tokenizer.encode_plus(text, max_length=batch_max_len,
                                                          padding='max_length',
                                                          return_tensors='pt', truncation=True)
                batch_input_ids.append(encoded_dict['input_ids'])
                batch_att_mask.append(encoded_dict['attention_mask'])
            batch_input_ids = torch.cat(batch_input_ids)
            batch_att_mask = torch.cat(batch_att_mask)

            batch_input_ids, batch_att_mask = batch_input_ids.to(self.device), batch_att_mask.to(self.device)

            with torch.no_grad():
                if self.is_cur_onnx_model:
                    # # 使用onnx模型推理，调用session.run
                    best_paths = self.onnx_model.run(
                        ['best_paths'],  # 设置输出names
                        {'input_ids': batch_input_ids.cpu().numpy(), 'att_mask': batch_att_mask.cpu().numpy()}  # feed输入
                    )[0]
                    best_paths = torch.LongTensor(best_paths).to(self.device)
                else:
                    best_paths = self.model(batch_input_ids, batch_att_mask)
                for best_path, att_mask in zip(best_paths, batch_att_mask):
                    active_labels = best_path[att_mask == 1][1:-1]  # 去除pad部分、[CLS]和[SEP]部分
                    labels = [self.vocab.id2tag[label_id.item()] for label_id in active_labels]
                    batch_labels.append(labels)

        return batch_labels

    def get_bert_tokenizer(self):
        return self.model.bert_tokenizer

    def transform2onnx(self, fix_seq_len):
        """将模型转换为onnx模型"""
        # # 定义伪输入，让onnx做一遍推理，构建静态计算图
        dummy_inputs = torch.LongTensor([[i for i in range(fix_seq_len)]])
        dummy_att_masks = torch.LongTensor([[1 for _ in range(fix_seq_len)]])
        dummy_inputs, dummy_att_masks = dummy_inputs.to(self.device), dummy_att_masks.to(self.device)
        # # 将模型导出为onnx标准
        torch.onnx.export(
            self.model, (dummy_inputs, dummy_att_masks), '{}/model.onnx'.format(self.model_dir),
            input_names=['input_ids', 'att_mask'], output_names=['best_paths'],  # 设置model的输入输出
            dynamic_axes={'input_ids': [0], 'att_mask': [0], 'best_paths': [0]},  # 设置batch维度可变长
        )
        # # 通过InferenceSession，加载onnx模型
        self.onnx_model = InferenceSession('{}/model.onnx'.format(self.model_dir))
        self.onnx_model.get_modelmeta()
        # # 更新is_cur_onnx_model状态
        self.is_cur_onnx_model = True
