import unittest

from ner.model.idcnn.idcnn_trainer import IDCNNCRFTrainer
from ner.model.idcnn.idcnn_predictor import IDCNNPredictor


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.model_dir = './tests/test_data/idcnn-crf'

    def test_trainer(self):
        texts = [
            ['你', '好', '呀'],
            ['一', '马', '当', '先', '就', '是', '好'],
        ]
        labels = [
            ['B', 'E', 'S'],
            ['B', 'M', 'M', 'E', 'S', 'S', 'S'],
        ]
        trainer = IDCNNCRFTrainer(
            self.model_dir, filters=64, hidden_num=256, embedding_size=100, dropout_rate=0.3, learning_rate=1e-3
        )
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=50, max_len=10)

    def test_predictor(self):
        predictor = IDCNNPredictor(self.model_dir)
        texts = [
            ['你', '好', '呀'],
            ['一', '马', '当', '先', '就', '是', '好'],
        ]
        labels = predictor.predict(texts)
        print(labels)


if __name__ == '__main__':
    unittest.main()
