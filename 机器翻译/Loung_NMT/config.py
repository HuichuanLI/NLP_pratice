# —*- coding: utf-8 -*-

import argparse

def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_size', type=int, default=10, help="embedding size of word embedding")
    parser.add_argument("--epoch",type=int,default=200,help="epoch of training")
    parser.add_argument("--cuda",type=bool,default=True,help="whether use gpu")
    parser.add_argument("--gpu",type=int,default=2,help="whether use gpu")
    parser.add_argument("--learning_rate",type=float,default=0.001,help="learning rate during training")
    parser.add_argument("--batch_size",type=int,default=32,help="batch size during training")
    parser.add_argument("--seed",type=int,default=1,help="seed of random")
    parser.add_argument("--dropout_rate",type=float,default=0.2,help="seed of random")
    parser.add_argument("--score_f",type=str,default="concat",help="attention score function such as dot, general, concat")
    parser.add_argument("--attention_c",type=str,default="local_p",help="attention class such as global, local_m, local_p")
    parser.add_argument("--window_size",type=int,default=10,help="local attention window size")
    parser.add_argument("--reverse",type=bool,default=True,help="whether reverse the input sentence")
    parser.add_argument("--feed_input",type=bool,default=True,help="whether use feed input")
    return parser.parse_args(args=[])
