# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import AutoModelForMaskedLM
from transformers import BertTokenizer, BertModel

from dist import euclidean_dist

class ProtoNet(nn.Module):
    def __init__(self, bert_name, num_ways, num_shots, num_queries):
        super(ProtoNet, self).__init__()
        self.encoder = BertModel.from_pretrained(bert_name)
        self.dropout = nn.Dropout(0.1) # optional
        self.linear = nn.Linear(768, 256)

        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_queries = num_queries
    
    def forward(self, support_params, query_params):
        """ 前向计算，得到每个类别的原型向量
        Attention：Hugginface 中 transformer 的输入顺序是 input_id, attention_mask, token_type_ids
        """
        
        support_feats = self.encoder(support_params[0], support_params[1], support_params[2]).pooler_output
        support_feats = self.linear(self.dropout(support_feats)) # shape of [batch_size, num_shots, 256]
        
        # [狗-1 狗-2 狗-3 狗-4 狗-5 猫-1 猫-2，。。。。猪-5]
        query_feats = self.encoder(query_params[0], query_params[1], query_params[2]).pooler_output
        query_feats = self.linear(self.dropout(query_feats))

        # 计算原型向量
        # [n_ways, n_shots, dim] -> [n_ways, dim]
        protos = support_feats.view(self.num_ways, self.num_shots, 256).mean(1) 

        # 为每个 query 样本生成标签
        # 3-ways-k-shots任务，query set 每个类别有 5 个查询样本。
        # [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2]
        target_inds = torch.arange(0, self.num_ways).view(self.num_ways, 1, 1).expand(self.num_ways, self.num_queries, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        target_inds = target_inds.cuda()

        # 计算距离
        dists = euclidean_dist(query_feats, protos) # shape of [num_ways*num_queries,]
        
        # 计算损失 crossentropy loss
        log_p_y = F.log_softmax(-dists, dim=1).view(self.num_ways, self.num_queries, -1)
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
    
        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'y_hat': y_hat
            }
    