import random
import unittest

import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss

from ner.focal_loss import FocalLoss


class MyTestCase(unittest.TestCase):
    def test(self):
        print('test~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logits = Variable(torch.rand(20, 2) * random.randint(1, 10))  # seq_len * label_size
        labels = Variable(torch.rand(20).ge(0.1).long())  # seq_len

        fl = FocalLoss(gamma=0, alpha=None)(logits, labels)  # gamma=0, alpha=None时， focal loss等价于交叉熵
        ce = CrossEntropyLoss()(logits, labels)
        print(fl, ce)
        print(FocalLoss(gamma=1, alpha=[0.1, 0.9])(logits, labels))  # 设置gamma,设置alpha标签权重分布


if __name__ == '__main__':
    unittest.main()
