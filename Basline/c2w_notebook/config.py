# —*- coding: utf-8 -*-

import argparse

def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=bool, default=True, help="whether use gpu")
    parser.add_argument("--gpu", type=int, default=1, help="whether use gpu")
    parser.add_argument('--n_chars', type=int, default=512, help="number of characters")
    parser.add_argument("--char_embed_size",type=int,default=50,help="character embedding size")
    parser.add_argument("--max_word_length",type=int,default=16,help="max number of characters in word")
    parser.add_argument("--max_sentence_length",type=int,default=100,help="max number of words in sentence")
    parser.add_argument("--char_hidden_size",type=int,default=150,help="hidden size of char lstm")
    parser.add_argument("--lm_hidden_size",type=int,default=150,help="hidden size of lm lstm")
    parser.add_argument("--word_embed_size",type=int,default=50,help="word embedding size")
    parser.add_argument("--vocab_size",type=int,default=5000,help="number of words")
    parser.add_argument("--learning_rate",type=float,default=0.0005,help="learning rate during training")
    parser.add_argument("--batch_size",type=int,default=200,help="batch size during training")
    parser.add_argument("--seed",type=int,default=1,help="seed of random")
    parser.add_argument("--epoch",type=int,default=100,help="epoch of training")
    return parser.parse_args(args=[])
