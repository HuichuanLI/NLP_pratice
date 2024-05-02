from gensim.models import KeyedVectors
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = []
datas = open("./imdb/imdb-train.txt.ss",encoding="utf-8").read().splitlines()
datas = [data.split("		")[-1].split() for data in datas]
model = word2vec.Word2Vec(datas, min_count=5)
model.wv.save_word2vec_format('imdb.model', binary=True)
wvmodel = KeyedVectors.load_word2vec_format("imdb.model",binary=True)
print (wvmodel.get_vector("good"))