"""
针对验证集/测试集结果进行评估
调用metrics.EntityScore

确保文件格式如下：

贾	B-PER.NAM	O
老	I-PER.NAM	O
板	I-PER.NAM	O
翻	O	O
唱	O	O
歌	O	O
曲	O	O
中	B-GPE.NAM	B-GPE.NAM
国	I-GPE.NAM	I-GPE.NAM
原	O	O
创	O	O
音	O	O
乐	O	O
基	O	O
地	O	O

第二列为真实标签、第三列为预测标签
"""

import sys
import json

from metrics import EntityScore

dev_result_f = sys.argv[1]  # 读取dev结果文件

with open(dev_result_f, 'r', encoding='utf-8', errors='ignore') as f:
    predict_tags, actual_tags = [], []
    for example in f.read().split('\n\n'):  # 迭代每条样本
        example = example.strip()
        if not example:
            continue
        predict_tags.append([]), actual_tags.append([])
        for term in example.split('\n'):
            if len(term.split('\t')) != 3:  # 跳过不合法的行
                continue
            char, actual_label, predict_label = term.split('\t')
            predict_tags[-1].append(predict_label), actual_tags[-1].append(actual_label)
    r = EntityScore.multi_entities_score(predict_tags, actual_tags)
    print(json.dumps(r, indent=4))
