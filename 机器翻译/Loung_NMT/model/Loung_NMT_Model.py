# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from .ScoreF import score_dot, score_concat, score_generate, local_m, local_p, local_score
class Loung_NMT(nn.Module):
    def __init__(self,source_vocab_size,target_vocab_size,embedding_size,
                 lstm_size, score_f, attention_c, feed_input, window_size=10, reverse=True):
        super(Loung_NMT,self).__init__()
        self.score_f = score_f
        self.attention_c = attention_c
        self.window_size = window_size
        self.feed_input = feed_input
        self.reverse = reverse
        self.source_embedding =nn.Embedding(source_vocab_size,embedding_size)
        self.target_embedding = nn.Embedding(target_vocab_size,embedding_size)
        self.encoder = nn.LSTM(input_size=embedding_size,hidden_size=lstm_size,num_layers=1,
                               batch_first=True)
        if not feed_input:
            self.decoder = nn.LSTM(input_size=embedding_size, hidden_size=lstm_size, num_layers=1,
                                   batch_first=True)
        else:
            self.decoder = nn.LSTM(input_size=embedding_size+lstm_size,hidden_size=lstm_size,num_layers=1,
                                   batch_first=True)
        self.class_fc_1 = nn.Linear(lstm_size+lstm_size, lstm_size) # 分类全连接层1
        self.class_fc_2 = nn.Linear(lstm_size, target_vocab_size) # 分类全连接层2

        self.attention_fc_1 = nn.Linear(lstm_size, lstm_size)
        self.attention_fc_2 = nn.Linear(2*lstm_size, lstm_size)
        self.attention_fc_3 = nn.Linear(lstm_size, 1)

        self.local_fc_1 = nn.Linear(lstm_size, lstm_size)
        self.local_fc_2 = nn.Linear(lstm_size, 1)

    def attention_forward(self,input_embedding, feed_input_h, dec_prev_hidden, enc_output, t):
        if not self.feed_input:
            dec_lstm_input = input_embedding
        else:
            dec_lstm_input = torch.cat([input_embedding, feed_input_h], dim=2)
        dec_output, dec_hidden  = self.decoder(dec_lstm_input, dec_prev_hidden)
        if self.score_f == "dot":
            attention_weights = score_dot(enc_output, dec_hidden[0].squeeze(), self.attention_fc_1, self.attention_fc_2, self.attention_fc_3)
        elif self.score_f == "general":
            attention_weights = score_generate(enc_output, dec_hidden[0].squeeze(), self.attention_fc_1, self.attention_fc_2, self.attention_fc_3)
        elif self.score_f == "concat":
            attention_weights = score_concat(enc_output, dec_hidden[0].squeeze(), self.attention_fc_1, self.attention_fc_2, self.attention_fc_3)
        else:
            print ("Attention score function input error!")
            exit()
        if self.attention_c == "local_m":
            if self.reverse:
                t = enc_output.size()[1]-1-t
            pt = local_m(dec_hidden[0].squeeze(), t)
            attention_weights = local_score(attention_weights, pt, enc_output.size()[1], self.window_size/2)
        elif self.attention_c == "local_p":
            pt = local_p(dec_hidden[0].squeeze(), t, self.local_fc_1, self.local_fc_2, enc_output.size()[1])
            attention_weights = local_score(attention_weights, pt, enc_output.size()[1], self.window_size / 2)
        elif self.attention_c == "global":
            pass
        else:
            print ("Attention class input error!")
            exit()
        atten_output = torch.sum(attention_weights * enc_output, dim=1).unsqueeze(1)
        return atten_output,dec_output,dec_hidden
    def forward(self, source_data,target_data, mode = "train",is_gpu=True):
        source_data_embedding = self.source_embedding(source_data)
        enc_output, enc_hidden = self.encoder(source_data_embedding)
        self.atten_outputs = Variable(torch.zeros(target_data.shape[0],
                                                  target_data.shape[1],
                                                  enc_output.shape[2]))
        self.dec_outputs = Variable(torch.zeros(target_data.shape[0],
                                                target_data.shape[1],
                                                enc_hidden[0].shape[2]))
        if is_gpu:
            self.atten_outputs = self.atten_outputs.cuda()
            self.dec_outputs = self.dec_outputs.cuda()
        # enc_output: bs*length*(2*lstm_size)
        if mode=="train":
            target_data_embedding = self.target_embedding(target_data)
            dec_prev_hidden = [enc_hidden[0],enc_hidden[1]]
            # dec_prev_hidden[0]: 1*bs*lstm_size, dec_prev_hidden[1]: 1*bs*lstm_size
            # dec_h: bs*lstm_size
            feed_input_h = enc_hidden[0].squeeze(0).unsqueeze(1)
            for i in range(100):
                input_embedding = target_data_embedding[:,i,:].unsqueeze(1)
                atten_output, dec_output, dec_hidden = self.attention_forward(input_embedding,
                                                                              feed_input_h,
                                                                              dec_prev_hidden,
                                                                              enc_output, i)
                self.atten_outputs[:,i] = atten_output.squeeze()
                self.dec_outputs[:,i] = dec_output.squeeze()
                dec_prev_hidden = dec_hidden
                feed_input_h = F.tanh(self.class_fc_1(torch.cat([atten_output,dec_output],dim=2)))
            outs = self.class_fc_2(F.tanh(self.class_fc_1(torch.cat([self.atten_outputs,self.dec_outputs],dim=2))))
        else:
            input_embedding = self.target_embedding(target_data)
            dec_prev_hidden = [enc_hidden[0], enc_hidden[1]]
            outs = []
            feed_input_h = enc_hidden[0].squeeze(0).unsqueeze(1)
            for i in range(100):
                atten_output, dec_output, dec_hidden = self.attention_forward(input_embedding,
                                                                              feed_input_h,
                                                                              dec_prev_hidden,
                                                                              enc_output, i)

                feed_input_h = F.tanh(self.class_fc_1(torch.cat([atten_output,dec_output],dim=2)))
                pred = self.class_fc_2(feed_input_h)
                pred = torch.argmax(pred,dim=-1)
                outs.append(pred.squeeze().cpu().numpy())
                dec_prev_hidden = dec_hidden
                input_embedding = self.target_embedding(pred)
        return outs