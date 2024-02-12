import unittest

from keywords.topic_model import TopicModel


class TestKeyword(unittest.TestCase):

    def setUp(self):
        self.topic_model = TopicModel()
        self.seg_data_file = './tests/test_data/sports_1000_seg.txt'
        self.lda_model_dir = './tests/test_data/lda'

    def test_lda(self):
        print('test_train_lda~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        self.topic_model.train_lda_model(seg_files=[self.seg_data_file], output_model_dir=self.lda_model_dir)

        print('test_extract_keywords~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        words = ['曝', 'NBA', '巨星', '来华', '路', '并未', '堵死', '山西', '签', '科比', '仍留', '生机']
        self.topic_model.load_lda_model(self.lda_model_dir)
        result = self.topic_model.extract_keywords(words)
        print(result)


if __name__ == '__main__':
    unittest.main()
