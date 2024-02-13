import torch
import torch.nn as nn


class IDCNN(nn.Module):
    """
    IDCNN模型

    原始论文中，整个模型由4个block组成，每个block由3个1维卷积组成，膨胀宽度dilation分别为1 1 2

Sequential(
  (block0): Sequential(
    (layer0): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
    (relu): ReLU()
    (layernorm): LayerNorm()
    (layer1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
    (layer2): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
  )
  (relu): ReLU()
  (layernorm): LayerNorm()
  (block1): Sequential(
    (layer0): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
    (relu): ReLU()
    (layernorm): LayerNorm()
    (layer1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
    (layer2): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
  )
  (block2): Sequential(
    (layer0): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
    (relu): ReLU()
    (layernorm): LayerNorm()
    (layer1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
    (layer2): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
  )
  (block3): Sequential(
    (layer0): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
    (relu): ReLU()
    (layernorm): LayerNorm()
    (layer1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
    (layer2): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
  )
)
)
    """

    def __init__(self, seq_len, embedding_size, filters, kernel_size=3, num_block=4):
        super(IDCNN, self).__init__()

        # 定义linear层，将embedding_size映射为filters
        # （主要统一多个block的输入输出）
        self.linear = nn.Linear(embedding_size, filters)

        # 定义单个block
        block = nn.Sequential()
        self.dilation_widths = [1, 1, 2]
        norms_1 = nn.ModuleList([LayerNorm(seq_len) for _ in range(len(self.dilation_widths))])
        for i in range(len(self.dilation_widths)):
            dilation = self.dilation_widths[i]
            # 设置卷积输入输出通道均为filters、设置kernel_size dilation padding
            conv = nn.Conv1d(in_channels=filters,
                             out_channels=filters,
                             kernel_size=kernel_size,
                             dilation=dilation,
                             padding=kernel_size // 2 + dilation - 1)
            block.add_module("layer%d" % i, conv)
            block.add_module("relu", nn.ReLU())
            block.add_module("layernorm", norms_1[i])

        # 组合4个block
        self.idcnn = nn.Sequential()
        norms_2 = nn.ModuleList([LayerNorm(seq_len) for _ in range(num_block)])
        for i in range(num_block):
            self.idcnn.add_module("block%d" % i, block)
            self.idcnn.add_module("relu", nn.ReLU())
            self.idcnn.add_module("layernorm", norms_2[i])

    def forward(self, embeddings, length):
        # 对embedding的输入做线性层变换
        embeddings = self.linear(embeddings)  # (batch_size, seq_len, embedding_size) => (batch_size, seq_len, filters)

        # 对embeddings切换坐标
        # （torch默认在最后一个维度进行一维卷积，这里确保最后一个维度是seq_len）
        embeddings = embeddings.permute(0, 2, 1)  # (batch_size, seq_len, filters) => (batch_size, filters, seq_len)

        # 输入idcnn层计算，并将输出维度切回
        output = self.idcnn(embeddings).permute(0, 2, 1)
        return output


class LayerNorm(nn.Module):
    """
    LayerNorm
    对每个样本做标准化（均值为0，方差为1），加速收敛
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
