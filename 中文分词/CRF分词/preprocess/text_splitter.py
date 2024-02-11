import math
import re


class TextSplitter(object):
    """
    文本分句
    分词应该在子句中进行（词不可能跨句）
    """

    def __init__(self, stops='([，。？?！!；;：\n 　])'):
        """
        通过正则，设置分句的分隔符

        [] 里面设置分隔符，表示或者的意思；()表示split的时候，返回分隔符，主要防止索引错位
        """
        self.stops = stops
        self.re_split_sentence = re.compile(self.stops)  # 提前编译正则，加速

    def split_sentence_for_seg(self, content, max_len=512):
        """
        使用正则regex.split分句

        限制子句的最大长度max_len（防止异常数据对分词器产生较大影响）
        """
        sentences = []

        for sent in self.re_split_sentence.split(content):  # 分句
            if not sent:  # 跳过空的句子
                continue
            for i in range(math.ceil(len(sent) / max_len)):  # 对子句进行max_len分段
                sent_segment = sent[i * max_len:(i + 1) * max_len]
                sentences.append(sent_segment)

        return sentences


if __name__ == '__main__':
    text_splitter = TextSplitter()
    content = '我是谁。我在哪里。你又是谁？119.2 。29,220.20元！你说：”我很好！是吗?”'
    print('split sentence for seg\n')
    for i, sent in enumerate(text_splitter.split_sentence_for_seg(content)):
        print('sentence num:{} sentence:{}'.format(i, sent))
    print('\n')
