# coding=utf-8

import os
import torch
from transformers import InputExample, AdamW, get_linear_schedule_with_warmup, \
    BertForMaskedLM, BertConfig, BertTokenizer, AutoTokenizer, AutoModelForMaskedLM

def get_data(data_dir, file_path, prefix, max_seq_length, tokenizer):
    sents = []
    labels = []
    with open(os.path.join(data_dir, file_path), 'r') as in_file:
        lines = in_file.readlines()
        for line in lines:
            sent, label = line.strip().split('\t')
            sents.append(prefix + sent)
            labels.append(int(label))
    
    inputs = tokenizer(sents, max_length=max_seq_length, padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor([label for label in labels], dtype=torch.long)
    
    return inputs, labels

def generate_template(num_prompt_tokens):
    """ 
    生成模板，并扩充 tokenizer。模板的形式多种多样，可根据需求设计模板生成函数。本代码仅用于生成如下前缀模板：
        '<prompt_1><prompt_2>[MASK] <prompt_3>...<prompt_n-2><prompt_n-1><prompt_n>'
    
    Args:
        num_prompt_tokens:
    
    Returns:
        tokenizer:
        template:
    """
    assert num_prompt_tokens >= 2
    prefix_template = '<prompt_1><prompt_2>[MASK]'

    for i in range(3, num_prompt_tokens+1):
        prefix_template = prefix_template + '<prompt_{}>'.format(i)

    return prefix_template

def extend_tokenizer(num_prompt_tokens, tokenizer):
    """
    将 prompt tokens 加入到分词器中，方便做预处理

    Args:
        num_prompt_tokens: 
    
    Returns:
        tokenizer:
    """
    prompt_tokens = []
    for i in range(1, num_prompt_tokens+1):
        token = '<prompt_{}>'.format(i)
        prompt_tokens.append(token)

    tokenizer.add_special_tokens({"additional_special_tokens":prompt_tokens})

    return tokenizer

def test_get_data():
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    prefix = '这家宾馆很[MASK]，'
    max_seq_length = 256
    inputs, labels = get_data('./data', 'train.tsv',  prefix, max_seq_length, tokenizer)
    print(inputs['input_ids'].size())

def test_generate_template():
    print(generate_template(6))

def test_extend_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    print(len(tokenizer))
    tokenizer = extend_tokenizer(6, tokenizer)
    print(len(tokenizer))
    print(tokenizer('<prompt_1><prompt_2>[MASK]')['input_ids'])

if __name__ == '__main__':
    test_generate_template()