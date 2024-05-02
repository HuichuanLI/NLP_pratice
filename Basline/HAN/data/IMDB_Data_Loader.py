#coding:utf-8
from torch.utils import data
import os
import torch
import nltk
import numpy as np
# from gensim.models import KeyedVectors
import nltk
class IMDB_Data(data.DataLoader):
    def __init__(self,data_name,min_count,word2id = None,max_sentence_length = 100,batch_size=64,is_pretrain=False):
        self.path = os.path.abspath(".")
        if "data" not in self.path:
            self.path += "/data"
        self.data_name = "/imdb/"+data_name
        self.min_count = min_count
        self.word2id = word2id
        self.max_sentence_length = max_sentence_length
        self.batch_size = batch_size
        self.datas,self.labels= self.load_data()
        if is_pretrain:
            self.get_word2vec()
        else:
            self.weight=None
        for i in range(len(self.datas)):
            self.datas[i] = np.array(self.datas[i])

    def load_data(self):
        datas = open(self.path+self.data_name,encoding="utf-8").read().splitlines()
        datas = [data.split("		")[-1].split()+[data.split("		")[2]] for data in datas]
        datas = sorted(datas,key = lambda x:len(x),reverse=True)
        labels  = [int(data[-1])-1 for data in datas]
        datas = [data[0:-1] for data in datas]
        if self.word2id ==None:
            self.get_word2id(datas)
        for i,data in enumerate(datas):
            datas[i] = " ".join(data).split("<sssss>")
            for j,sentence in enumerate(datas[i]):
                datas[i][j] = sentence.split()
        datas = self.convert_data2id(datas)
        return datas,labels
    def get_word2id(self,datas):
        word_freq = {}
        for data in datas:
            for word in data:
                word_freq[word] = word_freq.get(word,0)+1
        word2id = {"<pad>":0,"<unk>":1}
        for word in word_freq:
            if word_freq[word]<self.min_count:
                continue
            else:
                word2id[word] = len(word2id)
        self.word2id = word2id
    def convert_data2id(self,datas):
        for i,document in enumerate(datas):
            if i%10000==0:
                print (i,len(datas))
            for j,sentence in enumerate(document):
                for k,word in enumerate(sentence):
                    datas[i][j][k] = self.word2id.get(word,self.word2id["<unk>"])
                datas[i][j] = datas[i][j][0:self.max_sentence_length] + \
                              [self.word2id["<pad>"]]*(self.max_sentence_length-len(datas[i][j]))
        for i in range(0,len(datas),self.batch_size):
            max_data_length = max([len(x) for x in datas[i:i+self.batch_size]])
            for j in range(i,min(i+self.batch_size,len(datas))):
                datas[j] = datas[j] + [[self.word2id["<pad>"]]*self.max_sentence_length]*(max_data_length-len(datas[j]))
                datas[j] = datas[j]
        return datas

    '''def get_word2vec(self):
        print("Reading word2vec Embedding...")
        wvmodel = KeyedVectors.load_word2vec_format(self.path + "/imdb.model",binary=True)
        tmp = []
        for word, index in self.word2id.items():
            try:
                tmp.append(wvmodel.get_vector(word))
            except:
                pass
        mean = np.mean(np.array(tmp))
        std = np.std(np.array(tmp))
        print(mean, std)
        vocab_size = len(self.word2id)
        embed_size = 200
        embedding_weights = np.random.normal(mean, std, [vocab_size, embed_size])  # 正太分布初始化方法
        for word, index in self.word2id.items():
            try:
                embedding_weights[index, :] = wvmodel.get_vector(word)
            except:
                pass
        self.weight = torch.from_numpy(embedding_weights).float()'''

    def __getitem__(self, idx):
        return self.datas[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)
if __name__=="__main__":
    imdb_data = IMDB_Data(data_name="imdb-train.txt.ss",min_count=5,is_pretrain=True)
    training_iter = torch.utils.data.DataLoader(dataset=imdb_data,
                                                batch_size=64,
                                                shuffle=False,
                                                num_workers=0)
    for data, label in training_iter:
        print (np.array(data).shape)
