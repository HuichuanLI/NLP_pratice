import torch
from transformers import BertTokenizer, BertModel

"""
分类或序列标注任务，
单句子输入bert演示
"""

# 定义两个text
texts = [
    '你好呀',
    '我不好啊，good2013',
]

# 加载bert分词器（tokenizer）
# 这里可以选择不同的bert预训练模型
# chinese-bert-wwm和bert-base-chinese没有太大区别，具体区别后面讲
# （如果你已下载好bert-base-chinese，可以换成bert-base-chinese）
bert_tokenizer = BertTokenizer.from_pretrained('./model/chinese-bert-wwm')

# 获得bert的输入，input_ids和att_mask
# input_ids存放id形式的文本，att_mask非pad部分为1，否则为0
batch_input_ids, batch_att_mask = [], []
for text in texts:  # 迭代每个样本
    encoded_dict = bert_tokenizer.encode_plus(  # tokenizer得到id形式的tensor，并pad
        text, max_length=10, padding='max_length', return_tensors='pt', truncation=True
    )
    batch_input_ids.append(encoded_dict['input_ids'])  # 中文按照字被变成id，英文和数字按照词变成id（此时序列长度发生变化）
    batch_att_mask.append(encoded_dict['attention_mask'])
# 将一个batch文本变成tensor
batch_input_ids = torch.cat(batch_input_ids)
batch_att_mask = torch.cat(batch_att_mask)

# 加载bert模型
bert_model = BertModel.from_pretrained('./model/chinese-bert-wwm')

# 推理，查看bert输出
with torch.no_grad():
    output = bert_model(input_ids=batch_input_ids, attention_mask=batch_att_mask)
    print(output.last_hidden_state)
    # print('last_hidden_state', last_hidden_state, last_hidden_state.size)  # 可以接序列标注
    # print('\n')
    # print('pooled_output', pooled_output, pooled_output.size)  # 可以接分类
