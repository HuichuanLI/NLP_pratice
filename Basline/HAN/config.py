# —*- coding: utf-8 -*-

import argparse

def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_size', type=int, default=10, help="embedding size of word embedding")
    parser.add_argument("--epoch",type=int,default=200,help="epoch of training")
    parser.add_argument("--cuda",type=bool,default=True,help="whether use gpu")
    parser.add_argument("--gpu",type=int,default=1,help="whether use gpu")
    parser.add_argument("--learning_rate",type=float,default=0.001,help="learning rate during training")
    parser.add_argument("--batch_size",type=int,default=64,help="batch size during training")
    parser.add_argument("--seed",type=int,default=1,help="seed of random")
    parser.add_argument("--min_count",type=int,default=5,help="min count of words")
    parser.add_argument("--max_sentence_length",type=int,default=100,help="max sentence length")
    parser.add_argument("--embedding_size",type=int,default=200,help="word embedding size")
    parser.add_argument("--gru_size",type=int,default=50,help="gru size")
    parser.add_argument("--class_num",type=int,default=10,help="class num")
    return parser.parse_args(args=[])
