import math
import re


class TextSplitter(object):

    def __init__(self, stops='([，。？?！!；;：\n ])'):  # 注意，空格被视为分句标识，因此请确保输入不要有无意义的空格
        self.stops = stops
        self.re_split_sentence = re.compile(self.stops)

    def split_sentence_for_seg(self, content, max_len=512):
        """
        专为seg使用的分句

        理论单个子句会比较短，这里限制子句的max_len，考虑后面接入的比如bert模型
        """
        sentences = []

        for sent in self.re_split_sentence.split(content):
            if not sent:
                continue
            for i in range(math.ceil(len(sent) / max_len)):
                sent_segment = sent[i * max_len:(i + 1) * max_len]
                sentences.append(sent_segment)

        return sentences

    def split_sentence(self, content):
        """普通分句"""
        sentences = []
        tmp_sentences = [x for x in self.re_split_sentence.split(content) if x]
        for tmp_sent in tmp_sentences:
            if not sentences:
                sentences.append(tmp_sent)
            elif tmp_sent in self.stops and sentences[-1][-1] not in self.stops:
                sentences[-1] += tmp_sent
            else:
                sentences.append(tmp_sent)
        return sentences

    def split_sentence_merge_by_len(self, content, max_len):
        """按照stops分句，并根据最大长度合并句子"""
        sentences = []

        sent_buff = ''
        for sentence in self.split_sentence(content):
            for i in range(math.ceil(len(sentence) / max_len)):
                sent = sentence[i * max_len:(i + 1) * max_len]
                if len(sent_buff + sent) >= max_len:
                    if sent_buff:
                        sentences.append(sent_buff)
                        sent_buff = ''
                sent_buff += sent
        if sent_buff:
            sentences.append(sent_buff)

        return sentences
