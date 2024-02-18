import numpy as np
import torch
from sklearn.metrics import accuracy_score

from classification.bert_fc.bert_fc_predictor import BertFCPredictor
from classification.bert_fc.bert_fc_trainer import BertFCTrainer

# 设置随机种子
seed = 0
torch.manual_seed(seed)  # torch cpu随机种子
torch.cuda.manual_seed_all(seed)  # torch gpu随机种子
np.random.seed(seed)  # numpy随机种子


def read_data(data_path):
    """
    读取原始数据，返回titles、labels
    """
    titles, labels = [], []
    with open(data_path, 'r', encoding='utf-8') as f:
        print('current file:', data_path)
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            _, _, label, title, _ = line.split('_!_')
            titles.append(list(title)), labels.append([label])
        print(data_path, 'finish')
    return titles, labels


# 读取train dev test数据
train_path, dev_path, test_path = \
    './data/toutiao_cat_data.train.txt', './data/toutiao_cat_data.dev.txt', './data/toutiao_cat_data.test.txt'
(train_texts, train_labels), (dev_texts, dev_labels), (test_texts, test_labels) = \
    read_data(train_path), read_data(dev_path), read_data(test_path)

# 实例化trainer，设置参数，训练
pretrained_model = './model/bert-distil-chinese'  # # # 换预训练模型为distilbert
# pretrained_model = './model/chinese-roberta-wwm-ext'

trainer = BertFCTrainer(
    pretrained_model_dir=pretrained_model, model_dir='./tmp/bertfc', learning_rate=5e-5,
    enable_parallel=True,
    loss_type='cross_entropy_loss',
)
trainer.train(
    train_texts, train_labels, validate_texts=dev_texts, validate_labels=dev_labels, batch_size=64, epoch=4
)

# 实例化predictor，加载模型
predictor = BertFCPredictor(
    pretrained_model_dir=pretrained_model, model_dir='./tmp/bertfc',
    enable_parallel=True
)
predict_labels = predictor.predict(test_texts, batch_size=64)

# 评估
test_acc = accuracy_score(test_labels, predict_labels)
print('test acc:', test_acc)
