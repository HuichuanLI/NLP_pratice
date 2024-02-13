import json
import math

import torch
from sklearn.utils import shuffle
from torch import LongTensor
from torch.optim import Adam

from ner import logger
from ner.model.base.base_trainer import BaseTrainer
from ner.model.idcnn.idcnn_crf import IDCNN_CRF
from ner.model.vocab import Vocab


class IDCNNCRFTrainer(BaseTrainer):
    def __init__(self, model_dir, filters, hidden_num, embedding_size, dropout_rate, learning_rate=1e-3,
                 ckpt_name='idcnn-crf-model.bin', vocab_name='vocab.json', load_last_ckpt=True):
        self.model_dir = model_dir
        self.ckpt_name = ckpt_name
        self.vocab_name = vocab_name
        self.load_last_ckpt = load_last_ckpt

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 获取当前设备

        # 设置当前训练参数
        self.filters = filters
        self.hidden_num = hidden_num
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = None
        self.epoch = None
        self.max_len = None

        # 实例化空的vocab
        self.vocab = Vocab()

    def _build_model(self):
        """构建idcnn-crf模型"""
        # 初始化idcnn-crf模型
        self.model = IDCNN_CRF(
            seq_len=self.max_len, filters=self.filters, hidden_num=self.hidden_num, vocab_size=self.vocab.vocab_size,
            embedding_size=self.embedding_size, label_size=self.vocab.label_size, dropout_rate=self.dropout_rate
        )
        if self.load_last_ckpt:
            # 启用增量训练，预加载之前训练好的模型
            self.model.load_state_dict(
                torch.load('{}/{}'.format(self.model_dir, self.ckpt_name), map_location=self.device)
            )
        # 设置adam优化器
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        # 将模型拷贝到当前设备
        self.model.to(self.device)

    def _save_config(self):
        """将当前的训练配置保存为json"""
        config = {
            'max_len': self.max_len,
            'filters': self.filters,
            'hidden_num': self.hidden_num,
            'embedding_size': self.embedding_size,
            'dropout_rate': self.dropout_rate,
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

    def _transform_batch(self, batch_texts, batch_labels, max_length):
        """将batch的文本及labels转换为id的tensor形式"""
        batch_input_ids, batch_label_ids, input_lens = [], [], []
        for text, labels in zip(batch_texts, batch_labels):  # 迭代每条text, labels
            batch_input_ids.append([]), batch_label_ids.append([])
            assert len(text) == len(labels)  # 确保等长
            input_lens.append(len(labels))  # 更新input_lens
            for seg, label in zip(text, labels):  # 迭代每个位置的char和label
                # 通过vocab，将char和label都转化为id
                batch_input_ids[-1].append(self.vocab.vocab2id.get(seg, self.vocab.unk_vocab_id))
                batch_label_ids[-1].append(self.vocab.tag2id[label])
            # pad到max_length
            batch_input_ids[-1] += [self.vocab.pad_vocab_id] * (max_length - len(batch_input_ids[-1]))
            batch_label_ids[-1] += [self.vocab.pad_tag_id] * (max_length - len(batch_label_ids[-1]))

        # 转为tensor
        batch_input_ids, batch_label_ids, input_lens = \
            LongTensor(batch_input_ids), LongTensor(batch_label_ids), LongTensor(input_lens)

        # 将数据拷贝到当前设备
        batch_input_ids, batch_label_ids, input_lens = \
            batch_input_ids.to(self.device), batch_label_ids.to(self.device), input_lens.to(self.device)

        return batch_input_ids, batch_label_ids, input_lens

    def train(self, train_texts, labels, validate_texts, validate_labels, batch_size=30, epoch=10, max_len=200):
        """训练

        Args:
            train_texts: list[list[str]].训练样本
            labels: list[list[str]].标签
            validate_texts: list[list[str]].验证样本
            validate_labels: list[list[str]].验证集标签
            batch_size: int
            epoch: int
            max_len:
        """
        # 将train函数的一些参数更新到对象
        self.batch_size = batch_size
        self.epoch = epoch
        self.max_len = max_len

        self.vocab.build_vocab(texts=train_texts, labels=labels)  # 构建词库
        self._build_model()  # 构建模型
        # 保存词库和config
        self.vocab.save_vocab('{}/{}'.format(self.model_dir, self.vocab_name))
        self._save_config()

        logger.info('train samples: {}, validate samples: {}'.format(len(train_texts), len(validate_texts)))

        best_loss = float("inf")
        loss_buff = []  # 缓存最近的10个valid loss
        max_loss_num = 10  # 按照最新10个valid平均loss保存模型
        step = 0

        for epoch in range(epoch):  # 迭代每一轮
            for batch_idx in range(math.ceil(len(train_texts) / batch_size)):  # 迭代每一个batch
                text_batch = train_texts[batch_size * batch_idx: batch_size * (batch_idx + 1)]
                labels_batch = labels[batch_size * batch_idx: batch_size * (batch_idx + 1)]

                step = step + 1
                self.model.train()  # 设置训练模式
                self.model.zero_grad()  # 清空梯度

                # 转换batch原始文本和label为id的tensor形式
                batch_input_ids, batch_label_ids, input_lens = self._transform_batch(text_batch,
                                                                                     labels_batch,
                                                                                     self.max_len)
                # 输入模型，返回loss和预测结果
                best_path, loss = self.model(batch_input_ids, input_lens, labels=batch_label_ids)
                # 反向传播梯度，更新参数
                loss.backward()
                self.optimizer.step()

                # 统计训练集acc、验证集acc和loss
                train_acc = self._get_acc_one_step(best_path, batch_label_ids, input_lens)
                valid_acc, valid_loss = self.validate(validate_texts, validate_labels, sample_size=batch_size)

                # 更新最新的10个valid loss，更新avg loss
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

                # 发现当前avg loss最小，保存模型
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
            sample_size: int.采样大小(使用全量验证集可能较慢，这里每次随机采样sample_size个样本做验证)

        Returns:
            float.验证集上acc, loss
        """
        # 设置评估模式（预测模式）
        self.model.eval()
        # 随机采样sample_size个样本
        batch_texts, batch_labels = [
            return_val[:sample_size] for return_val in shuffle(validate_texts, validate_labels)
        ]
        # 计算valid acc, valid loss
        with torch.no_grad():
            batch_input_ids, batch_label_ids, input_lens = self._transform_batch(batch_texts,
                                                                                 batch_labels,
                                                                                 self.max_len)
            best_path, loss = self.model(batch_input_ids, input_lens, labels=batch_label_ids)
            acc = self._get_acc_one_step(best_path, batch_label_ids, input_lens)
            return acc, loss

    def _get_acc_one_step(self, best_path, labels_batch, input_lengths):
        """
        计算一个batch的acc
        使用input_lengths，忽略掉padding的部分
        """
        total, correct = 0, 0  # 总共的标签数，正确的标签数
        for predict_labels, labels, input_length in zip(best_path, labels_batch, input_lengths.tolist()):
            total += input_length  # 根据input_length，更新总标签数
            correct += (predict_labels[:input_length] == labels[:input_length].cpu()).int().sum().item()  # 更新正确标签数
        accuracy = correct / total  # 计算acc
        return float(accuracy)
