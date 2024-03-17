# —*- coding: utf-8 -*-

import argparse

def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_size', type=int, default=300, help="embedding size of word embedding")
    parser.add_argument("--epoch",type=int,default=1,help="epoch of training")
    parser.add_argument("--cuda",type=bool,default=True,help="whether use gpu")
    parser.add_argument("--gpu",type=int,default=0,help="whether use gpu")
    parser.add_argument("--learning_rate",type=float,default=0.001,help="learning rate during training")
    parser.add_argument("--batch_size",type=int,default=32,help="batch size during training")
    parser.add_argument("--min_count",type=int,default=20,help="min count of words")
    parser.add_argument("--window_size",type=int,default=2,help="min count of words")
    parser.add_argument("--x_max",type=int,default=100,help="x_max of glove")
    parser.add_argument("--alpha",type=float,default=0.75,help="alpha of glove")

    return parser.parse_args(args=[])
