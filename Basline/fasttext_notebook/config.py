# —*- coding: utf-8 -*-

import argparse

def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_size', type=int, default=10, help="embedding size of word embedding")
    parser.add_argument("--epoch",type=int,default=200,help="epoch of training")
    parser.add_argument("--cuda",type=bool,default=True,help="whether use gpu")
    parser.add_argument("--gpu",type=int,default=1,help="whether use gpu")
    parser.add_argument("--label_num",type=int,default=4,help="the label number of samples")
    parser.add_argument("--learning_rate",type=float,default=0.02,help="learning rate during training")
    parser.add_argument("--batch_size",type=int,default=64,help="batch size during training")
    parser.add_argument("--seed",type=int,default=1,help="seed of random")
    parser.add_argument("--max_length",type=int,default=100,help="max length of sentence")
    parser.add_argument("--min_count",type=int,default=3,help="min_count of word")
    parser.add_argument("--n_gram",type=int,default=2,help="num gram of word")
    parser.embedding_pretrained = None
    return parser.parse_args(args=[])
