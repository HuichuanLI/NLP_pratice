"""
处理数据

使用jieba分词，标注新闻语料
（机器标注，可能存在不少错误；实际业务场景中，可以让业务方校对下机器标注结果，提升标注质量）
"""
import jieba
from preprocess.text_splitter import TextSplitter
from preprocess import text_normalize


def _test_jieba_dl_cut():
    """
    测试jieba的深度学习分词
    """
    content = '正在意大利度蜜月的“脸谱”创始人扎克伯格与他华裔妻子的一举一动都处于媒体的追踪之下'
    # jieba.enable_paddle()
    words = jieba.cut(content)
    print(' '.join(words))


def read_raw_data(f_name='data/news_tensite_xml.smarty.dat'):
    """
    读取原始搜狐新闻数据，返回干净的正文
    """
    with open(f_name, 'r', encoding='gb18030') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '<content>' in line and '</content>' in line:  # 只使用content这行
                content = line.replace('<content>', '').replace('</content>', '').strip()  # 去除content标签
                if not content:  # 跳过空的content
                    continue
                yield content


def gen_train_data():
    """
    将文本分句、全角转半角，
    生成两列数据：
        今   B
        天   E
        很   S
        开   B
        心   E
    """
    text_splitter = TextSplitter()
    # jieba.enable_paddle()  # 启用paddle的深度学习分词
    out = open('data/train_data.txt', 'w', encoding='utf-8')

    for line in read_raw_data():
        for sent in text_splitter.split_sentence_for_seg(line):
            if sent in text_splitter.stops[2:-2]:  # 去除仅包含标点符号的句子
                continue
            sent = text_normalize.string_q2b(sent)  # 全角转半角

            # 生成标注标签
            labels = []  # 序列标注的标签
            words = jieba.cut(sent)
            for word in words:
                if len(word) == 1:
                    labels.append('S')
                else:
                    labels += ['B'] + ['M'] * (len(word) - 2) + ['E']

            # 写文件
            out.write(
                '\n'.join([
                    '{}\t{}'.format(char, label) for char, label in zip(sent, labels)
                ])
            )
            out.write('\n\n')  # 两句话中间多一个空行

    out.close()


if __name__ == '__main__':
    _test_jieba_dl_cut()
    # print('\n'.join(read_raw_data()))
    # gen_train_data()
