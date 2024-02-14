import evaluate
from ner.models.bert_crf.bert_crf_predictor import BERTCRFPredictor
from ner.models.bert_crf.bert_crf_trainer import BertCRFTrainer


def read_conll(file_path):
    with open(file_path, encoding='utf-8', errors='ignore') as f:
        texts, labels = [], []
        for example in f.read().split('\n\n'):  # 迭代每条样本
            example = example.strip()
            if not example:
                continue
            texts.append([]), labels.append([])
            for term in example.split('\n'):
                if len(term.split('\t')) != 2:  # 跳过不合法的行
                    continue
                char, label = term.split('\t')
                texts[-1].append(char), labels[-1].append(label)
        return texts, labels


# 读取数据
train_texts, train_labels = read_conll('./data/weiboNER.conll.train')
dev_texts, dev_labels = read_conll('./data/weiboNER.conll.dev')
test_texts, test_labels = read_conll('./data/weiboNER.conll.test')

# 实例化trainer，设置参数，训练
trainer = BertCRFTrainer(
    pretrained_model_dir='./model/chinese-bert-wwm', model_dir='./tmp/bercrf', learning_rate=1e-4
)
trainer.train(
    train_texts, train_labels, validate_texts=dev_texts, validate_labels=dev_labels, batch_size=4, epoch=5
)

# 实例化predictor，加载模型
predictor = BERTCRFPredictor(pretrained_model_dir='./model/chinese-bert-wwm', model_dir='./tmp/bercrf')
predict_labels = predictor.predict(test_texts, batch_size=20)

# 将结果输出为3列
out = open('tmp/dev_results.txt', 'w', encoding='utf-8')
for text, each_true_labels, each_predict_labels in zip(test_texts, test_labels, predict_labels):
    for char, true_label, predict_label in zip(text, each_true_labels, each_predict_labels):
        out.write('{}\t{}\t{}\n'.format(char, true_label, predict_label))
    out.write('\n')
out.close()

# 评估
evaluate.eval('tmp/dev_results.txt')
