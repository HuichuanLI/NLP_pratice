import unittest

from keywords.keyword import Keyword


class TestKeyword(unittest.TestCase):

    def setUp(self):
        self.keyword = Keyword('./tests/test_data/idf.txt')

    def test_textrank(self):
        print('测试textrank方法')
        title = ''
        content = '新化县人民法院院长程海辉严重渎职官商勾结'
        for top_k in range(5):
            keywords = self.keyword.extract(title=title, content=content, method='textrank', with_weight=False, top_k=2)
            print(keywords)
            keywords = self.keyword.extract(title=title, content=content, method='textrank', with_weight=True)
            print(keywords)

    def test_tfidf(self):
        print('测试tfidf方法')
        title = ''
        content = '新化县人民法院院长程海辉严重渎职官商勾结'
        for top_k in range(5):
            keywords = self.keyword.extract(title=title, content=content, method='TFIDF', with_weight=False, top_k=2)
            print(keywords)
            keywords = self.keyword.extract(title=title, content=content, method='TFIDF', with_weight=True)
            print(keywords)


if __name__ == '__main__':
    unittest.main()
