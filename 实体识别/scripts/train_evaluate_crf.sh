cd ..
# 设置参数，使用crf_learn训练，保存模型
crf_learn -f 1 -c 1 -a CRF-L2 data/template_3_3.txt data/weiboNER.conll.train tmp/crf-model.bin

# 使用刚刚训练好的模型，在train集、dev集上预测，保存到tmp目录
crf_test -m tmp/crf-model.bin data/weiboNER.conll.train > ./tmp/train_result.txt
crf_test -m tmp/crf-model.bin data/weiboNER.conll.dev > ./tmp/dev_result.txt


# 统计train集、dev集上指标
echo "train metric:"
python evaluate.py ./tmp/train_result.txt
echo "\ndev metric:"
python evaluate.py ./tmp/dev_result.txt
