import unittest

from ner.ner_dict import NerDict


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._dict_path = './test_data/word_dict.txt'

    def test1(self):
        print('\ntest加词删词~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        content = '张三和李四都在北京大学玩'
        dict = NerDict(self._dict_path)
        print('初始结果: {}'.format(dict.findall(content)))
        print('添加PER: 李四')
        dict.add_word('李四', 'PER')
        print('添加ORG: 北京大学')
        dict.add_word('北京大学', 'ORG')
        print('新结果: {}'.format(dict.findall(content)))
        print('删除ORG: 北京大学')
        dict.delete_word('北京大学', 'ORG')
        print('新结果: {}'.format(dict.findall(content)))

    def test2(self):
        print('\ntest_findall~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        content = '张三和李四都在北京大学玩'
        dict = NerDict(self._dict_path)
        dict.add_word('北京大学', 'ORG')
        print('init: {}'.format(dict.findall(content)))  # 默认参数
        print('types=ORG, {}'.format(dict.findall(content, types=('ORG',))))  # 只返回ORG
        print('with_offset=False, {}'.format(dict.findall(content, with_offset=False)))  # 不返回索引


if __name__ == '__main__':
    unittest.main()
