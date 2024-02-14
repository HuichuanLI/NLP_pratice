import unittest

from ner.model.bert_crf.bert_crf_predictor import BERTCRFPredictor
from ner.model.bert_crf.bert_crf_trainer import BertCRFTrainer


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.model_dir = './tests/test_model/bertcrf'
        self.pretrained_model_dir = './model/chinese-bert-wwm'

    def test_trainer(self):
        texts = [
            ['你', '好', '呀'],
            ['一', '马', '当', '先', '就', '是', '好'],
        ]
        labels = [
            ['B', 'E', 'S'],
            ['B', 'M', 'M', 'E', 'S', 'S', 'S']
        ]
        trainer = BertCRFTrainer(self.pretrained_model_dir, self.model_dir, learning_rate=1e-4)
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=10)

    def test_predictor(self):
        predictor = BERTCRFPredictor(self.pretrained_model_dir, self.model_dir)
        texts = [
            ['你', '好', '呀'],
            ['一', '马', '当', '先', '就', '是', '好'],
        ]
        labels = predictor.predict(texts)
        print(labels)


if __name__ == '__main__':
    unittest.main()
