# coding=utf-8

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from p_tuning import SoftPromptWrapper, PromptEncoder
from utils import generate_template, extend_tokenizer, get_data

from transformers import InputExample, AdamW, get_linear_schedule_with_warmup, \
    BertForMaskedLM, BertConfig, BertTokenizer, AutoTokenizer, AutoModelForMaskedLM
from torch.optim import AdamW

import argparse
import numpy as np
from sklearn.metrics import classification_report

# Global 
mask_idx = 3 # 模板 <prompt_1><prompt_2>[MASK]<prompt_3>...， mask index 一直为3（注意 [CLS]）

verbalizer = {
    0: '差',
    1: '好'
}

model_name = 'bert-base-chinese'
max_seq_length = 256

# 最佳结果 8； 5e-5；32 token

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

    parser.add_argument("--device",
                        default='cuda:0',
                        type=str,
                        help="device to train model")

    parser.add_argument("--num_prompt_tokens",
                        default=8,
                        type=int,
                        help="number of prompt tokens")

    parser.add_argument("--epochs",
                        default=40,
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
                        default=5e-5,
                        type=int,
                        help="learning rate")
    args = parser.parse_args()

    
    device = torch.device(args.device)

    # 初始化
    tokenizer = BertTokenizer.from_pretrained(model_name)

    bert_mlm = BertForMaskedLM.from_pretrained(model_name)
    prompt_embedding = PromptEncoder(args.num_prompt_tokens, len(tokenizer)) # 21128

    prefix = generate_template(args.num_prompt_tokens)
    tokenizer = extend_tokenizer(args.num_prompt_tokens, tokenizer) # 21134

    model = SoftPromptWrapper(bert_mlm, prompt_embedding)
    model.to(device)
    model.zero_grad()

    neg_token_id = tokenizer.convert_tokens_to_ids(verbalizer[0])
    pos_token_id = tokenizer.convert_tokens_to_ids(verbalizer[1])

    loss_func = CrossEntropyLoss()
    # optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    optimizer = AdamW(model.prompt_embdding.parameters(), lr=args.learning_rate)
    for p in model.underlying_model.parameters():
        p.requires_grad = False

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
            print(loss.cpu().item())

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