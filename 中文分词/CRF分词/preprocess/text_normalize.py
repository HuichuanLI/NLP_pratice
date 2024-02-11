def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if '\u4e00' <= uchar <= '\u9fa5':
        return True
    else:
        return False


def b2q(uchar):
    """半角转全角"""
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e:  # 不是半角字符就返回原来的字符
        return uchar
    if inside_code == 0x0020:  # 除了空格其他的全角半角的公式为:半角=全角-0xfee0
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return chr(inside_code)


def q2b(uchar):
    """全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:  # 空格
        inside_code = 0x0020
    else:  # 半角=全角-0xfee0
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)


def string_q2b(ustring):
    """把字符串全角转半角"""
    return ''.join([q2b(uchar) for uchar in ustring])


if __name__ == '__main__':
    s = '天气不错！是么？'
    print(string_q2b(s))
