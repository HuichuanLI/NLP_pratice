# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule


class TextCNN(BasicModule):

    def __init__(self, config):
        super(TextCNN, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed_size)
        if config.cuda:
            self.convs = [nn.Conv1d(config.embed_size, config.filter_num, filter_size).cuda()
                          for filter_size in config.filters]
        else:
            self.convs = [nn.Conv1d(config.embed_size, config.filter_num, filter_size)
                          for filter_size in config.filters]
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.filter_num*len(config.filters), config.label_num)
    def conv_and_pool(self,x,conv):
        x = F.relu(conv(x))
        x = F.max_pool1d(x,x.size(2)).squeeze(2)
        return x
    def forward(self, x):
        out = self.embedding(x)
        out = out.transpose(1, 2).contiguous()
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    print('running the TextCNN...')