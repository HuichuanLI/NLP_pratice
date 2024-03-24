# —*- coding: utf-8 -*-

import argparse

def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch",type=int,default=200,help="epoch of training")
    parser.add_argument("--cuda",type=bool,default=True,help="whether use gpu")
    parser.add_argument("--gpu",type=int,default=0,help="whether use gpu")
    parser.add_argument("--label_num",type=int,default=4,help="the label number of samples")
    parser.add_argument("--learning_rate",type=float,default=0.0001,help="learning rate during training")
    parser.add_argument("--batch_size",type=int,default=50,help="batch size during training")
    parser.add_argument("--char_num",type=int,default=70,help="character number of samples")
    parser.add_argument("--features",type=str,default="256,256,256,256,256,256",help="filters size of conv")
    parser.add_argument("--kernel_sizes",type=str,default="7,7,3,3,3,3",help="kernel size of conv")
    parser.add_argument("--pooling",type=str,default="1,1,0,0,0,1",help="is use pooling of convs")
    parser.add_argument("--l0",type=int,default="1014",help="length of character sentence")
    parser.add_argument("--dropout",type=float,default=0.5,help="dropout of training")
    parser.add_argument("--num_classes",type=int,default=4,help="number classes of data")
    parser.add_argument("--seed",type=int,default=1,help="seed of random")
    return parser.parse_args(args=[])
