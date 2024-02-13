import unittest

import jieba

from keywords.textrank import TextRank


class Test(unittest.TestCase):
    def setUp(self) -> None:
        # 加载停用词
        self.stop_words = set()
        with open('stop_words.txt', 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if not word:
                    continue
                self.stop_words.add(word)

    def test_textrank(self):
        print('测试textrank方法~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        content = '报道称，拜登政府对中美经济关系的全面审查已进行了7个月的时间，现在审查应当回答一个核心问题，即如何处理前总统特朗普在2020年初签署的协议。在这份协议中，为了保护美国的关键产业，如汽车和飞机制造，特朗普政府威胁对价值3600亿美元的中国进口商品开征关税，经过双方协商，中国承诺购买更多美国产品并改变其贸易做法。但拜登政府至今没有提及这笔交易将何去何从，导致价值3600 亿美元的中国进口商品的关税悬而未决。'
        textrank = TextRank()
        words = [word for word in jieba.cut(content) if word not in self.stop_words]  # 去除停用词
        keywords = textrank.textrank(words, window_size=3)
        print(keywords)


if __name__ == '__main__':
    unittest.main()
