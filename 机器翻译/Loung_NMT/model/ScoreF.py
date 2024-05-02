# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable

def score_dot(H_e, h_d, fc_layer_1=None, fc_layer_2=None, fc_layer_3=None):
    '''
    Get attention score throught dot function
    :param H_e: encoder hiddens as batch_size * source_length * hidden_size
    :param h_d:decoder hidden as batch_size * hidden_size
    :return: attention vector as batch_size * hidden_size
    '''
    h_d = h_d.unsqueeze(2)
    attention_score = torch.matmul(H_e, h_d) # batch_size * source_length * 1
    attention_score = F.softmax(attention_score, dim=1)
    return attention_score

def score_generate(H_e, h_d,fc_layer_1=None, fc_layer_2=None, fc_layer_3=None):
    H_e = fc_layer_1(H_e)
    attention_score = score_dot(H_e, h_d)
    return attention_score

def score_concat(H_e, h_d, fc_layer_1=None, fc_layer_2=None, fc_layer_3=None):
    h_d = h_d.unsqueeze(1).repeat([1, H_e.size()[1], 1])

    attention_score = fc_layer_3(F.tanh(fc_layer_2(torch.cat([H_e, h_d], dim=-1))))
    attention_score = F.softmax(attention_score, dim=1)
    return attention_score

def local_m(h_d, t):
    pt = torch.ones([h_d.size()[0],1]) * t
    pt = pt.cuda()
    return pt
def local_p(h_d, t, fc_layer_1, fc_layer_2, seq_len):
    pt = seq_len * F.sigmoid(fc_layer_2(F.tanh(fc_layer_1(h_d))))
    return pt
def local_score(attention_score, pt, seq_len, sigma):
    pt = pt.unsqueeze(2)
    s = torch.range(0, seq_len-1).cuda()
    s = s.view([1, seq_len, 1]).repeat([attention_score.size()[0],1,1])
    attention_score = attention_score * torch.exp(-(s - pt)**2/(2*sigma**2))
    attention_score = attention_score/torch.sum(attention_score, dim=1, keepdim=True)
    return attention_score


