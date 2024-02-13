import os
import unittest

from keywords.tfidf import TFIDF


class TestKeyword(unittest.TestCase):

    def setUp(self):
        self.tfidf = TFIDF()
        self.seg_data_dir = 'tests/test_data/seg_data'
        self.idf_file_path = './tests/test_data/idf.txt'

    def test_tf_idf(self):
        print('test_train_idf~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        seg_files = ['{}/{}'.format(self.seg_data_dir, f) for f in os.listdir(self.seg_data_dir)]
        self.tfidf.train_idf(seg_files=seg_files,
                             output_file_name=self.idf_file_path)
        print('test_tfidf~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        words = ['我', '发现', '院长', '渎职', '我', '很', '惊讶']
        self.tfidf.load_idf(self.idf_file_path)
        result = self.tfidf.compute_tfidf(words)
        print(result)


if __name__ == '__main__':
    unittest.main()
