import unittest

from classification.bert_fc.bert_fc_predictor import BertFCPredictor
from classification.bert_fc.bert_fc_trainer import BertFCTrainer


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.model_dir = './tests/test_data'
        self.pretrained_model_dir = '/Users/brown/Downloads/chinese-roberta-wwm-ext'

    def test_trainer(self):
        texts = [
            ['天', '气', '真', '好'],
            ['今', '天', '运', '气', '很', '差'],
        ]
        labels = [
            ['正面'],  # 每个label是一个长度为1的list，主要和序列标注、多标签分类保持一致
            ['负面']
        ]
        trainer = BertFCTrainer(self.pretrained_model_dir, self.model_dir, learning_rate=5e-5)
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=10)

    def test_predictor(self):
        predictor = BertFCPredictor(self.pretrained_model_dir, self.model_dir)
        texts = [
            ['天', '气', '真', '好'],
            ['今', '天', '运', '气', '很', '差'],
            ['天', '气', '好', '极', '了'],
            ['运', '气', '贼', '差'],
        ]
        labels = predictor.predict(texts)
        print(labels)

    def test_adv_trainer(self):
        texts = [
            ['天', '气', '真', '好'],
            ['今', '天', '运', '气', '很', '差'],
        ]
        labels = [
            ['正面'],
            ['负面']
        ]
        trainer = BertFCTrainer(self.pretrained_model_dir, self.model_dir, learning_rate=5e-5, adversarial='fgm')
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=10)


if __name__ == '__main__':
    unittest.main()
