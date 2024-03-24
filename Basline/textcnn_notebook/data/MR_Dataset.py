#coding:utf-8
from torch.utils import data
import os
import random
import numpy as np
import nltk
import torch
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

class MR_Dataset(data.Dataset):
    def __init__(self,state="train",k=0):

        self.path = os.path.abspath('.')
        if "data" not in self.path:
            self.path+="/data"
        pos_samples = open(self.path+"/MR/rt-polarity.pos",errors="ignore").readlines()
        neg_samples = open(self.path+"/MR/rt-polarity.neg",errors="ignore").readlines()
        datas = pos_samples+neg_samples
        #datas = [nltk.word_tokenize(data) for data in datas]
        datas = [data.split() for data in datas]
        max_sample_length = max([len(sample) for sample in datas])
        labels = [1]*len(pos_samples)+[0]*len(neg_samples)
        word2id = {"<pad>":0}
        for i,data in enumerate(datas):
            for j,word in enumerate(data):
                if word2id.get(word)==None:
                    word2id[word] = len(word2id)
                datas[i][j] = word2id[word]
            datas[i] = datas[i]+[0]*(max_sample_length-len(datas[i]))
        self.n_vocab = len(word2id)
        self.word2id = word2id
        #self.get_glove_embedding()
        self.get_word2vec()
        c = list(zip(datas,labels))
        random.seed(1)
        random.shuffle(c)
        datas[:],labels[:] = zip(*c)
        if state=="train":
            self.datas = datas[:int(k * len(datas) / 10)] + datas[int((k + 1) * len(datas) / 10):]
            self.labels = labels[:int(k * len(datas) / 10)] + labels[int((k + 1) * len(labels) / 10):]
            self.datas = np.array(self.datas[0:int(0.9*len(self.datas))])
            self.labels = np.array(self.labels[0:int(0.9*len(self.labels))])
        elif state == "valid":
            self.datas = datas[:int(k * len(datas) / 10)] + datas[int((k + 1) * len(datas) / 10):]
            self.labels = labels[:int(k * len(datas) / 10)] + labels[int((k + 1) * len(labels) / 10):]
            self.datas = np.array(self.datas[int(0.9 * len(self.datas)):])
            self.labels = np.array(self.labels[int(0.9 * len(self.labels)):])
        elif state == "test":
            self.datas = np.array(datas[int(k * len(datas) / 10):int((k + 1) * len(datas) / 10)])
            self.labels = np.array(labels[int(k * len(datas) / 10):int((k + 1) * len(datas) / 10)])
    def __getitem__(self, index):
        return self.datas[index], self.labels[index]

    def __len__(self):
        return len(self.datas)
    def get_glove_embedding(self):
        if not os.path.exists(self.path+"/glove_embedding_mr.npy"):
            if not os.path.exists(self.path+"/test_word2vec.txt"):
                glove_file = datapath(self.path+'/glove.840B.300d.txt')
                # 指定转化为word2vec格式后文件的位置
                tmp_file = get_tmpfile(self.path+"/glove_word2vec.txt")
                from gensim.scripts.glove2word2vec import glove2word2vec
                glove2word2vec(glove_file, tmp_file)
            else:
                tmp_file = get_tmpfile(self.path+"/glove_word2vec.txt")
            print ("Reading Glove Embedding...")
            wvmodel = KeyedVectors.load_word2vec_format(tmp_file)
            vocab_size = self.n_vocab
            embed_size = 300
            embedding_weights = np.random.randn(vocab_size,embed_size)
            for word, index in self.word2id.items():
                try:
                    embedding_weights[index, :] = wvmodel.get_vector(word)
                except:
                    pass
            np.save(self.path+"/glove_embedding_mr.npy", embedding_weights)
        else:
            embedding_weights = np.load(self.path+"/glove_embedding_mr.npy")
        self.weight = embedding_weights
    def get_word2vec(self):
        if  not os.path.exists(self.path+"/word2vec_embedding_mr.npy"):
            print ("Reading word2vec Embedding...")
            wvmodel = KeyedVectors.load_word2vec_format(self.path+"/GoogleNews-vectors-negative300.bin.gz",binary=True)
            tmp = []
            for word, index in self.word2id.items():
                try:
                    tmp.append(wvmodel.get_vector(word))
                except:
                    pass
            mean = np.mean(np.array(tmp))
            std = np.std(np.array(tmp))
            print (mean,std)
            vocab_size = self.n_vocab
            embed_size = 300
            #embedding_weights = np.random.normal(-0.0016728516,0.17756976,[vocab_size,embed_size])
            embedding_weights = np.random.normal(mean,std,[vocab_size,embed_size])
            for word, index in self.word2id.items():
                try:
                    embedding_weights[index, :] = wvmodel.get_vector(word)
                except:
                    pass
            np.save(self.path+"/word2vec_embedding_mr.npy", embedding_weights)
        else:
            embedding_weights = np.load(self.path+"/word2vec_embedding_mr.npy")
        self.weight = embedding_weights


if __name__=="__main__":
    mr_train_dataset = MR_Dataset()
    print (mr_train_dataset.__len__())
    print (mr_train_dataset[0])
    mr_valid_dataset = MR_Dataset("valid")
    print(mr_valid_dataset.__len__())
    print(mr_valid_dataset[0])
    mr_test_dataset = MR_Dataset("test")
    print(mr_test_dataset.__len__())
    print(mr_test_dataset[0])



