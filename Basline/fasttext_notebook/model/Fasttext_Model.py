# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Fasttext(nn.Module):
    def __init__(self,vocab_size,embedding_size,max_length,label_num):
        super(Fasttext,self).__init__()
        self.embedding =nn.Embedding(vocab_size,embedding_size)
        self.avg_pool = nn.AvgPool1d(kernel_size=max_length,stride=1)
        self.fc = nn.Linear(embedding_size, label_num)
    def forward(self, x):
        out = self.embedding(x) # batch_size*length*embedding_size
        out = out.transpose(1, 2).contiguous() # batch_size*embedding_size*length
        out = self.avg_pool(out).squeeze()
        out = self.fc(out) # batch_size*label_num
        return out
if __name__=="__main__":
    fasttext = Fasttext(100,200,100,4)
    x = torch.Tensor(np.zeros([64,100])).long()
    out = fasttext(x)
    print (out.size())