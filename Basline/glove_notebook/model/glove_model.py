# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
class glove_model(nn.Module):
    def __init__(self,vocab_size,embed_size,x_max,alpha):
        super(glove_model, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.x_max = x_max
        self.alpha = alpha
        self.w_embed = nn.Embedding(self.vocab_size,self.embed_size).type(torch.float64)

        self.w_bias = nn.Embedding(self.vocab_size,1).type(torch.float64)

        self.v_embed = nn.Embedding(self.vocab_size, self.embed_size).type(torch.float64)

        self.v_bias = nn.Embedding(self.vocab_size, 1).type(torch.float64)
    def forward(self, w_data,v_data,labels):
        w_data_embed = self.w_embed(w_data)
        w_data_bias = self.w_bias(w_data)
        v_data_embed = self.v_embed(v_data)
        v_data_bias = self.v_bias(v_data)
        weights = torch.pow(labels/self.x_max,self.alpha)
        weights[weights>1]=1
        loss = torch.mean(weights*torch.pow(torch.sum(w_data_embed*v_data_embed,1)+w_data_bias+v_data_bias-
                                 torch.log(labels),2))
        return loss
    def save_embedding(self, word2id, file_name):
        embedding_1 = self.w_embed.weight.data.cpu().numpy()
        embedding_2 = self.v_embed.weight.data.cpu().numpy()
        embedding = (embedding_1+embedding_2)/2
        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(word2id), self.embed_size))
        for w, wid in word2id.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))

