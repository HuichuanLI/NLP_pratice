# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
class Deep_NMT(nn.Module):
    def __init__(self,source_vocab_size,target_vocab_size,embedding_size,
                 source_length,target_length,lstm_size):
        super(Deep_NMT,self).__init__()
        self.source_embedding =nn.Embedding(source_vocab_size,embedding_size)
        self.target_embedding = nn.Embedding(target_vocab_size,embedding_size)
        self.encoder = nn.LSTM(input_size=embedding_size,hidden_size=lstm_size,num_layers=4,
                               batch_first=True)
        self.decoder = nn.LSTM(input_size=embedding_size,hidden_size=lstm_size,num_layers=4,
                               batch_first=True)
        self.fc = nn.Linear(lstm_size, target_vocab_size)
    def forward(self, source_data,target_data, mode = "train"):
        source_data_embedding = self.source_embedding(source_data)
        enc_output, enc_hidden = self.encoder(source_data_embedding)
        if mode=="train":
            target_data_embedding = self.target_embedding(target_data)

            dec_output, dec_hidden = self.decoder(target_data_embedding,enc_hidden)
            outs = self.fc(dec_output)
        else:
            target_data_embedding = self.target_embedding(target_data)
            dec_prev_hidden = enc_hidden
            outs = []
            for i in range(100):
                dec_output, dec_hidden = self.decoder(target_data_embedding, dec_prev_hidden)
                pred = self.fc(dec_output)
                pred = torch.argmax(pred,dim=-1)
                outs.append(pred.squeeze().cpu().numpy())
                dec_prev_hidden = dec_hidden
                target_data_embedding = self.target_embedding(pred)
        return outs
if __name__=="__main__":
    deep_nmt = Deep_NMT(source_vocab_size=30000,target_vocab_size=30000,embedding_size=256,
                 source_length=100,target_length=100,lstm_size=256)
    source_data = torch.Tensor(np.zeros([64,100])).long()
    target_data = torch.Tensor(np.zeros([64,100])).long()
    preds = deep_nmt(source_data,target_data)
    print (preds.shape)
    target_data = torch.Tensor(np.zeros([64, 1])).long()
    preds = deep_nmt(source_data, target_data,mode="test")
    print(np.array(preds).shape)