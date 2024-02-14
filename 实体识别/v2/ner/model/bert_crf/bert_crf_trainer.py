import json
import math

import torch
from sklearn.utils import shuffle
from transformers import AdamW

from ner import logger
from ner.model.base.base_trainer import BaseTrainer
from ner.model.bert_crf.bert_crf_model import BertCRFModel
from ner.model.vocab import Vocab


class BertCRFTrainer(BaseTrainer):
    def __init__(self, pretrained_model_dir, model_dir, learning_rate=1e-3, ckpt_name='bert_model.bin',
                 vocab_name='vocab.json'):
        self.pretrained_model_dir = pretrained_model_dir
        self.model_dir = model_dir
        self.ckpt_name = ckpt_name
        self.vocab_name = vocab_name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.learning_rate = learning_rate
        self.batch_size = None
        self.epoch = None

        self.vocab = Vocab()

    def _build_model(self):
        """构建bert-crf模型"""
        self.model = BertCRFModel(self.pretrained_model_dir, self.vocab.label_size)

        # 设置AdamW优化器
        no_decay = ["bias", "LayerNorm.weight"]  # bias和LayerNorm不使用正则化
        bert_parameters = self.model.named_parameters()
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        # 使用bert的vocab更新我们的Vocab对象
        self.vocab.set_vocab2id(self.model.get_bert_tokenizer().vocab)
        self.vocab.set_id2vocab({_id: char for char, _id in self.vocab.vocab2id.items()})
        self.vocab.set_unk_vocab_id(self.vocab.vocab2id['[UNK]'])
        self.vocab.set_pad_vocab_id(self.vocab.vocab2id['[PAD]'])

        self.model.to(self.device)

    def _save_config(self):
        config = {
            'vocab_size': self.vocab.vocab_size,
            'label_size': self.vocab.label_size,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epoch': self.epoch,
            'ckpt_name': self.ckpt_name,
            'vocab_name': self.vocab_name
        }
        with open('{}/train_config.json'.format(self.model_dir), 'w') as f:
            f.write(json.dumps(config, indent=4))

    def _transform_batch(self, batch_texts, batch_labels, max_length=512):
        """将batch的文本及labels转换为bert的输入tensor形式"""
        batch_input_ids, batch_att_mask, batch_label_ids = [], [], []
        for text, labels in zip(batch_texts, batch_labels):
            assert isinstance(text, list)
            text = ' '.join(text)  # 确保输入encode_plus函数为文本
            encoded_dict = self.model.bert_tokenizer.encode_plus(text, max_length=max_length, padding='max_length',
                                                                 return_tensors='pt', truncation=True)
            batch_input_ids.append(encoded_dict['input_ids'])
            batch_att_mask.append(encoded_dict['attention_mask'])
            batch_label_ids.append(
                [-1] + [self.vocab.tag2id[_label] for _label in labels] + [-1] +  # [CLS]、[SEP]用-1填充
                [self.vocab.pad_tag_id] * (max_length - len(labels) - 2)  # 尾部用pad填充
            )
        batch_input_ids = torch.cat(batch_input_ids)
        batch_att_mask = torch.cat(batch_att_mask)
        batch_label_ids = torch.LongTensor(batch_label_ids)

        batch_input_ids, batch_att_mask, batch_label_ids = \
            batch_input_ids.to(self.device), batch_att_mask.to(self.device), batch_label_ids.to(self.device)

        return batch_input_ids, batch_att_mask, batch_label_ids

    def train(self, train_texts, labels, validate_texts, validate_labels, batch_size=30, epoch=10):
        """训练

        Args:
            train_texts: list[list[str]].训练样本
            labels: list[list[str]].标签
            validate_texts: list[list[str]].验证样本
            validate_labels: list[list[str]].验证集标签
            batch_size: int
            epoch: int

        """
        self.batch_size = batch_size
        self.epoch = epoch

        self.vocab.build_vocab(labels=labels, build_texts=False)  # 只构建labels，词库bert已有
        self._build_model()
        self.vocab.save_vocab('{}/{}'.format(self.model_dir, self.vocab_name))
        self._save_config()

        logger.info('train samples: {}, validate samples: {}'.format(len(train_texts), len(validate_texts)))

        best_loss = float("inf")
        loss_buff = []  # 保存最近的10个valid loss
        max_loss_num = 10
        step = 0

        for epoch in range(epoch):
            for batch_idx in range(math.ceil(len(train_texts) / batch_size)):
                text_batch = train_texts[batch_size * batch_idx: batch_size * (batch_idx + 1)]
                labels_batch = labels[batch_size * batch_idx: batch_size * (batch_idx + 1)]

                step = step + 1
                self.model.train()
                self.model.zero_grad()

                batch_max_len = max([len(text) for text in text_batch]) + 2  # 长度得加上[CLS]和[SEP]
                batch_input_ids, batch_att_mask, batch_label_ids = self._transform_batch(text_batch,
                                                                                         labels_batch,
                                                                                         max_length=batch_max_len)
                logits, loss = self.model(batch_input_ids, batch_att_mask, labels=batch_label_ids)
                loss.backward()
                self.optimizer.step()

                train_acc = self._get_acc_one_step(logits, batch_label_ids)
                valid_acc, valid_loss = self.validate(validate_texts, validate_labels, sample_size=batch_size)
                loss_buff.append(valid_loss)
                if len(loss_buff) > max_loss_num:
                    loss_buff = loss_buff[-max_loss_num:]
                avg_loss = sum(loss_buff) / len(loss_buff) if len(loss_buff) == max_loss_num else None

                logger.info(
                    'epoch %d, step %d, train loss %.4f, train acc %.4f, valid loss %.4f valid acc %.4f, '
                    'last %d avg valid loss %s' % (
                        epoch, step, loss, train_acc, valid_loss, valid_acc, max_loss_num,
                        '%.4f' % avg_loss if avg_loss else avg_loss
                    )
                )

                if avg_loss and avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(self.model.state_dict(), '{}/{}'.format(self.model_dir, self.ckpt_name))
                    logger.info("model saved")

        logger.info("finished")

    def validate(self, validate_texts, validate_labels, sample_size=100):
        """使用当前的model评估验证集

        Args:
            validate_texts: list[list[str]] or np.array.验证样本（原始的或者转换过）
            validate_labels: list[list[str]] or np.array.验证集标签（原始的或者转换过）
            sample_size: int.采样大小(使用全量验证集较慢，这里每次随机采样sample_size个样本做验证)

        Returns:
            float.验证集上acc, loss
        """
        self.model.eval()
        # 随机采样sample_size个样本
        batch_texts, batch_labels = [
            return_val[:sample_size] for return_val in shuffle(validate_texts, validate_labels)
        ]
        # 计算valid acc, valid loss
        batch_max_len = max([len(text) for text in batch_texts]) + 2
        with torch.no_grad():
            batch_input_ids, batch_att_mask, batch_label_ids = self._transform_batch(batch_texts,
                                                                                     batch_labels,
                                                                                     max_length=batch_max_len)
            logits, loss = self.model(batch_input_ids, batch_att_mask, labels=batch_label_ids)
            acc = self._get_acc_one_step(logits, batch_label_ids)
            return acc, loss

    def _get_acc_one_step(self, logits, labels_batch):
        total, correct = 0, 0
        for logit, labels in zip(logits, labels_batch):
            active_logit = logit[labels != self.vocab.pad_tag_id][1:-1]  # 去除pad部分、[CLS]和[SEP]部分
            active_labels = labels[labels != self.vocab.pad_tag_id][1:-1]
            total += len(active_labels)
            correct += (torch.argmax(active_logit, dim=1) == active_labels).sum().item()
        accuracy = correct / total
        return float(accuracy)
