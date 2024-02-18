# coding=utf-8

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import InputExample, AdamW, get_linear_schedule_with_warmup, \
    BertForMaskedLM, BertConfig, BertTokenizer, AutoTokenizer, AutoModelForMaskedLM
from torch.optim import AdamW

import argparse
import numpy as np
from sklearn.metrics import classification_report

from utils import get_data


# Global 
# 根据要使用的预训练模型，构造 prompt。
# GPT：从左往右，'卫生很差' + '总之，这家宾馆很[MASK]'
prefix = '这家宾馆很[MASK]，' # huggingface tokenizer 会自动加上 [CLS]
mask_idx = 6

verbalizer = {
    0: '差',
    1: '好'
}

model_name = 'bert-base-chinese'
max_seq_length = 256


def train():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default='./data/',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--train_file_name",
                        default='train.tsv',
                        type=str,
                        help="train file name")
    parser.add_argument("--dev_file_name",
                        default='dev.tsv',
                        type=str,
                        help="dev file name")

    parser.add_argument("--epochs",
                        default=0,
                        type=int,
                        help="number of training epochs")
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="number of training epochs")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="number of training epochs")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=int,
                        help="learning rate")
    args = parser.parse_args()

    # 初始化
    device = torch.device('cuda:0')

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.zero_grad()

    neg_token_id = tokenizer.convert_tokens_to_ids(verbalizer[0]) # 差 token_id
    pos_token_id = tokenizer.convert_tokens_to_ids(verbalizer[1]) # 好 token_id
    

    loss_func = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)


    # 训练数据加载
    train_inputs, train_labels = get_data(args.data_dir, args.train_file_name, prefix, max_seq_length, tokenizer)

    train_data = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], \
        train_inputs['token_type_ids'], train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    # training stage
    for epoch in range(args.epochs):
        # 数据循环
        for input_ids, attention_masks, segment_ids, labels in train_dataloader:
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            segment_ids = segment_ids.to(device)
            labels = labels.to(device)

            lm_logits = model(input_ids, attention_masks, segment_ids).logits # [batch_size, seq_len, vocab_size]
            lm_logits = lm_logits[:, mask_idx] # [batch_size, vocab_size]
            cls_logits = lm_logits[:, [neg_token_id, pos_token_id]] # [batch_size, 2]
            
            loss = loss_func(cls_logits, labels)

            loss.backward()
            optimizer.step()
            model.zero_grad()
    
    # Eval stage
    model.eval()
    eval_inputs, eval_labels = get_data(args.data_dir, args.dev_file_name, prefix, max_seq_length, tokenizer)

    eval_data = TensorDataset(eval_inputs['input_ids'], eval_inputs['attention_mask'], \
        eval_inputs['token_type_ids'], eval_labels)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.train_batch_size)

    eval_preds = []
    eval_gts = []
    for input_ids, attention_masks, segment_ids, labels in eval_dataloader:
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            segment_ids = segment_ids.to(device)
            
            eval_gts.extend(labels.tolist())

            lm_logits = model(input_ids, attention_masks, segment_ids).logits # [batch_size, seq_len, vocab_size]
            lm_logits = lm_logits[:, mask_idx] # [batch_size, vocab_size]
            cls_logits = lm_logits[:, [neg_token_id, pos_token_id]] # [batch_size, 2]
            
            batch_preds = torch.argmax(cls_logits, dim=1)
            eval_preds.extend(batch_preds.cpu().tolist())

    print(classification_report(np.array(eval_gts), np.array(eval_preds)))

if __name__ == '__main__':
    train()