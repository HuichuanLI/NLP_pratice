import unittest

from ner.ner import NER


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._content = '九月三日上午，习近平、李克强、栗战书、汪洋、王沪宁、赵乐际、韩正、王岐山等党和国家领导人来到中国人民抗日战争纪念馆，' \
                        '出席纪念中国人民抗日战争暨世界反法西斯战争胜利七十五周年向抗战烈士敬献花篮仪式。'
        self._long_content = '[事件记录吧]【求文】求一篇古早狗血误会文,可用海棠换! 求个小可爱破案!第一本小说是一篇古早文,应该是10年左右的贴吧文,我依稀记得当年在文包里看到的时候是贴吧的格式,文包名字好像是误会文。情节就是攻的妹妹喜欢受,有一天约受出去但是出了事狗带了,攻情绪不稳定,攻的手下骗攻说是受害的他妹妹【其实并不是】于是攻就把受抓过来,用受的妹妹威胁攻,这里有一个情节是攻找了几个手下要轮受,受为了妹妹也答应了,但是攻又没让了,这个时候手下也松了口气说太好了我不是同这种话之后攻受就一直保持着床【伴】关系,受有一个妹妹在读贵族学校,有一天老师打电话过来问受说受的妹妹钢琴天赋很不错建议学钢琴,让受去学校谈一谈,路上攻也陪着受一起去了,谈到学钢琴的时候受说自己也学过,但因为手短一直谈不好,希望妹妹可以做自己想做的而不是被逼着学钢琴,这里攻有点触动后来见到妹妹的时候受问妹妹是不是决定要一直学钢琴了,妹妹想了想说自己还是更喜欢画画回去之后有一个场景是攻受在酒吧还是哪里,攻有事离开了,受被一些人缠住差点要出事了,结果攻过来了,但是以为是受勾引的那些人,于是对受动了手这个时候受已经是发烧了还是怎么,迷迷糊糊的,然后一直说不是自己,还说了你不信我,你们都不信我这种话,最后因为身体原因进了医院然后攻感觉不对重新查了妹妹的事,发现自己被骗了,他的手下x姐就说是因为要让攻振作起来所以找了一个替罪羊,然后攻开始忏悔说把手欺负的很惨balabala,好像还提到了受的肺有问题小说到这里大概是一半的内容 之后的情节就是受被攻的仇家绑架攻去救他啥,然后好像为了受哪里哪里受伤了,最后受回到了学校两个人在一起了'

    def test_merge(self):
        print('test_merge~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        model_r = {
            'PER': [(0, '张三和'), (3, '李四')],
        }
        dict_r = {
            'PER': [(0, '张三')],
            'ORG': [(7, '北京大学')]
        }
        print('result with offset: {}'.format(NER._merge_model_dict_result(model_r, dict_r)))
        print('result without offset: {}'.format(NER._merge_model_dict_result(model_r, dict_r, with_offset=False)))

    def test_find(self):
        print('test_find~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        ner = NER()
        print('enbale_offset True: {}'.format(ner.find(self._content, enable_offset=True)))
        print('long text: {}'.format(ner.find(self._long_content, enable_offset=True)))
        print('enbale_offset False: {}'.format(ner.find(self._content, enable_offset=False)))
        print('types=PER,ORG: {}'.format(ner.find(self._content, types=['PER', 'ORG'])))
        print('model=crf: {}'.format(ner.find(self._content, model='CRF')))

    def test_find_by_model(self):
        print('test_find_by_model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        ner = NER()
        r = ner.find_by_model(self._content, enable_offset=True)
        print(r)

    def test_find_by_dict(self):
        print('test_find_by_dict~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        ner = NER()
        ner.add_word('习近平', 'PER')
        ner.add_word('李克强', 'PER')
        r = ner.find_by_dict(self._content, enable_offset=True)
        print(r)

    def test_find_max_len(self):
        print('test_find_max_len~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        ner = NER()
        try:
            ner.find(self._content * 1000, enable_offset=True)
        except Exception as e:
            assert isinstance(e, AssertionError), '长度限制断言未触发！'


if __name__ == '__main__':
    unittest.main()
