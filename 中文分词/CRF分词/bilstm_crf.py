import torch
import torch.nn as nn
import torch.optim as optim
import json

torch.manual_seed(1)


def argmax(vec):
    """
    返回vec中最大值的下标，int
    """
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, word2id):
    """
    将文本序列转化为id形式的tensor
    """
    idxs = [word2id[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def log_sum_exp(vec):
    """
    log_sum_exp，即 log(e^x1 + e^x2 + e^x3 + ... + e^xn)

    这里以数值稳定的方式计算，即 c + log(e^(x1-c) + e^(x2-c) + ... + e^(xn-c))
    c为vec中最大值
    （这里不要太纠结）
    """
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        """
        :param vocab_size: 词库大小
        :param tag_to_ix: tag id映射
        :param embedding_dim: embedding维度
        :param hidden_dim: 隐含层维度
        """
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)  # 获得tag size

        # 定义embedding层，
        # 设置
        #   词库大小（输入序列的最大id不能超过它）
        #   embedding维度（一个词被表示为向量的维度）
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        # 定义lstm层，
        # 设置
        #   输入dim为embedding_dim
        #   隐含层单元为 1/2 hidden_dim（双向拼接之后为hidden_dim，送入下一层）
        #   一层lstm
        #   设置双向
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        # 将bilstm的输出通过全连接层映射到标签空间
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # CRF层的核心参数，转移矩阵
        # transitions[i,j]表示从j转移到i的score
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)
        )

        # 使不能出现 转移到START_TAG
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        # 使不能出现 从STOP_TAG转移
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        # self.hidden = self.init_lstm_hidden()

    def init_lstm_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        """
        前向算法
        从左往右，依次计算所有边的log_sum_exp
        """
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG包含所有的score
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        forward_var = init_alphas

        for feat in feats:  # 迭代整个句子所有时刻
            alphas_t = []  # 当前时刻所有状态的前向tensor
            for next_tag in range(self.tagset_size):  # 迭代每一个状态
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)

                # trans_score的第i项是从i转移到next_tag的分数
                trans_score = self.transitions[next_tag].view(1, -1)

                # next_tag_var的第i项是在进行log-sum-exp前边的值
                next_tag_var = forward_var + trans_score + emit_score

                # 当前标签的前向变量是对所有score的log-sum-exp
                alphas_t.append(log_sum_exp(next_tag_var).view(1))

            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        """
        根据输入文本，获取lstm的发射score矩阵
        """
        # 将sentence输入embedding层，并改变shape（torch中rnn，batch dim在中间件）
        # shape: (seq_len, batch, embedding_size)
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)

        # 随机初始化lstm隐含层参数
        self.hidden = self.init_lstm_hidden()
        # 将embeds输入bilstm，获得输出，并更新hidden
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # 更改lstm的shape，为(seq_len, hidden_dim)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)

        # 将lstm的输出接入全连接映射层，获得（篱笆网络）发射矩阵
        lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # 在对数空间初始化维特比变量
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # 第i步的 forward_var 存放第i-1步的维特比变量
        forward_var = init_vvars

        for feat in feats:
            bptrs_t = []  # 存放这一步的后指针
            viterbivars_t = []  # 存放这一步的维特比变量

            for next_tag in range(self.tagset_size):

                # next_tag_var[i] 存放先前一步标签i的维特比变量, 加上了从标签i到next_tag的转移
                # （这里暂时没有将发射score添加进来，因为最大值并不依赖它们。我们下面会加它。）
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            # 现在将所有发射score相加，更新forward_var
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # 转移到STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # 使用后指针解码最优路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # 弹出开始标签，无需返回它
        start = best_path.pop()

        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        """
        已知真实标签，计算crf loss

        log_sum_exp(所有路径score) - gold路径score
        """
        # 将sentence用bilstm编码，获得发射score矩阵
        feats = self._get_lstm_features(sentence)

        # 前向算法计算所有路径score的log_sum_exp
        forward_score = self._forward_alg(feats)

        # 计算真实路径的gold score
        gold_score = self._score_sentence(feats, tags)

        # 计算crf loss
        return forward_score - gold_score

    def forward(self, sentence):
        """
        前向阶段（预测解码阶段）
        """
        # 将sentence用bilstm编码，获得发射score矩阵
        lstm_feats = self._get_lstm_features(sentence)

        # 根据发射矩阵和转移矩阵，使用维特比算法计算最优标签路径
        score, tag_seq = self._viterbi_decode(lstm_feats)

        return score, tag_seq


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 128
HIDDEN_DIM = 100
EPOCH = 1

# 构造训练数据
training_data = []
blocks = open('data/train_data.txt', 'r', encoding='utf-8').read().split('\n\n')[:-1]
for block in blocks:
    chars, tags = [], []
    for line in block.split('\n'):
        char, tag = line.split('\t')
        chars.append(char)
        tags.append(tag)
    training_data.append((chars, tags))

# 构建word到id的映射
word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
word_to_ix['<UNK>'] = len(word_to_ix)  # 添加UNK字符

# 设置tag到id的映射
tag_to_ix = {"B": 0, "M": 1, "E": 2, "S": 3, START_TAG: 4, STOP_TAG: 5}
ix_to_tag = {ix: tag for tag, ix in tag_to_ix.items()}

# 初始化BiLSTM-CRF模型，传入必要的参数
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)

# 设置SGD优化器，设置lr
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# 训练前检验下前向/预测流程是否OK
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

# 训练
for epoch in range(EPOCH):  # 迭代多个epoch（完整的训过一次数据称为1轮，1个epoch）
    print('cur epoch', epoch)

    for i, (sentence, tags) in enumerate(training_data):  # 迭代每一个样本

        # pytorch会累计梯度，我们需要清理上一次计算的梯度
        model.zero_grad()

        # 准备输入文本和标签，转为下标的tensor
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # 前向传播，计算loss
        loss = model.neg_log_likelihood(sentence_in, targets)

        # 反向传播，计算梯度，更新参数
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('cur example', i)

# 保存模型、vocab、超参数
torch.save(model.state_dict(), 'model/torch-bilstm-crf-seg.bin')
with open('./model/vocab.json', 'w', encoding='utf-8') as f:
    json.dump({'word2id': word_to_ix, 'tag2id': tag_to_ix, 'id2tag': ix_to_tag}, f, ensure_ascii=False, indent=2)
with open('./model/params.json', 'w', encoding='utf-8') as f:
    json.dump({'embedding_dim': EMBEDDING_DIM, 'hidden_num': HIDDEN_DIM}, f, ensure_ascii=False, indent=2)

# 预测
with torch.no_grad():
    precheck_sent = prepare_sequence('今天天气不错哈', word_to_ix)
    score, tags = model(precheck_sent)
    tags = [ix_to_tag[tag] for tag in tags]
    print('score', score)
    print('tags', tags)
