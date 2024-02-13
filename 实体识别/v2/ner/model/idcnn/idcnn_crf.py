import torch
import torch.nn as nn

from ner.model.crf.crf_layer import CRF
from ner.model.idcnn.idcnn import IDCNN

torch.manual_seed(16)


class IDCNN_CRF(nn.Module):
    def __init__(self, seq_len, filters, hidden_num, vocab_size, embedding_size, label_size, dropout_rate):
        """
        :param seq_len: 固定的序列长度
        :param filters: 卷积核的个数
        :param hidden_num: 隐含层个数
        :param vocab_size: 词库大小
        :param embedding_size: embedding维度
        :param label_size: 标签个数
        :param dropout_rate: dropout比例
        """
        super(IDCNN_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # 定义idcnn层，设置seq_len、embedding_size、filters
        self.idcnn = IDCNN(seq_len=seq_len, embedding_size=embedding_size, filters=filters)
        # 定义dropout层
        self.dropout = nn.Dropout(dropout_rate)
        # 定义线性层，将上层输出映射到hidden层
        self.linear = nn.Linear(filters, hidden_num)
        # 定义线性层，将上层输出映射到各个label
        self.out = nn.Linear(hidden_num, label_size)
        # 定义CRF层
        self.crf = CRF(label_size)

    def forward(self, inputs, length, labels=None):
        # 逐层前向计算
        embeddings = self.embedding(inputs)
        embeddings = self.dropout(embeddings)
        out = self.idcnn(embeddings, length)
        out = self.linear(out)
        logits = self.out(out)
        # 根据logits，维特比计算最优标签路径
        best_path = self.crf.get_batch_best_path(logits, length)

        if labels is not None:
            # 如果有labels，计算crf_loss
            loss = self.crf.negative_log_loss(inputs=logits, length=length, tags=labels)
            return best_path, loss
        return best_path
