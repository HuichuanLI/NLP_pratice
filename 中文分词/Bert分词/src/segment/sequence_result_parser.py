class SequenceResultParser(object):
    """解析序列标注的BMES结果，生成chunk结果"""

    @staticmethod
    def parse(content, labels, with_offset=False, ignore_s=False):
        """

        Args:
            content: str or list.输入content
            labels: list.输入对应的标签
            with_offset: bool.是否返回索引
            ignore_s: bool.是否忽略标签为S的内容

        """
        assert len(content) == len(labels), '请确保content和labels长度相等！'

        temp_buf = []
        content = list(content)
        for idx, label in enumerate(labels):
            char = content[idx]
            label = label[0]
            if label == 'B':
                if temp_buf:
                    temp_buf = []
                temp_buf.append(char)
            elif label == 'M':
                if not temp_buf:
                    continue
                temp_buf.append(char)
            elif label == 'E':
                temp_buf.append(char)
                buff_str = ''.join(temp_buf)
                yield buff_str if not with_offset else (idx - len(buff_str) + 1, buff_str)
                temp_buf = []
            else:
                if temp_buf:  # 异常序列，清空缓存
                    buff_str = ''.join(temp_buf)
                    yield buff_str if not with_offset else (idx - len(buff_str), buff_str)
                    temp_buf = []
                if not ignore_s:
                    yield char if not with_offset else (idx, char)

    @staticmethod
    def parse_pos(content, labels, with_offset=False):
        """解析带词性的BMES结果

        Args:
            content:
            labels:
            with_offset:

        """
        assert len(content) == len(labels), '请确保content和labels长度相等！'

        temp_buf = []
        content = list(content)
        for idx, label in enumerate(labels):
            char = content[idx]
            if label[0] == 'B':
                if temp_buf:
                    temp_buf = []
                temp_buf.append(char)
            elif label[0] == 'M':
                if not temp_buf:
                    continue
                temp_buf.append(char)
            elif label[0] == 'E':
                temp_buf.append(char)
                buff_str = ''.join(temp_buf)
                yield (buff_str, label[2:]) if not with_offset else (idx - len(buff_str) + 1, buff_str, label[2:])
                temp_buf = []
            else:
                if temp_buf:  # 异常序列，清空缓存
                    buff_str = ''.join(temp_buf)
                    yield (buff_str, label[2:]) if not with_offset else (idx - len(buff_str), buff_str, label[2:])
                    temp_buf = []
