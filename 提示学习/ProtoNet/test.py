# coding=utf-8


import torch
from torch.optim import Adam
from transformers import BertTokenizer

import random
import argparse
import numpy as np
from collections import defaultdict

from model import ProtoNet
from utils import create_examples, convert_examples_to_features, load_data, random_extract_samples


def test():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--num_ways",
                        default=2,
                        type=int,
                        help="支撑集的类别数")
    parser.add_argument("--num_shots",
                        default=5,
                        type=int,
                        help="支撑集中每个类别的样本数量")
    parser.add_argument("--num_queries",
                        default=10,
                        type=int,
                        help="查询集中每个类别的样本数量")

    parser.add_argument("--bert_name",
                        default='bert-base-chinese',
                        type=str,
                        help="需要加载的 bert 模型名称")
    parser.add_argument("--device", 
                        default="cuda:0",
                        type=str,
                        help='训练所使用 GPU 卡')
    parser.add_argument("--seed", 
                        default=42,
                        type=int,
                        help='随机种子')
    parser.add_argument("--model_dir",
                        default='./output/model.pt',
                        type=str,
                        help="模型加载地址")
    parser.add_argument("--max_seq_length",
                        default=40,
                        type=int,
                        help="输入样本的最大长度")
    parser.add_argument("--test_file_path",
                        default='./data/test.txt',
                        type=str,
                        help="训练数据地址")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 数据集
    class2sents = load_data(args.test_file_path)

    # 模型初始化
    model = ProtoNet(args.bert_name, args.num_ways, args.num_shots, args.num_queries)
    model.load_state_dict(torch.load(args.model_dir))

    tokenizer = BertTokenizer.from_pretrained(args.bert_name)
    device = torch.device(args.device)
    model.to(device)

    # 模拟测试阶段
    model.eval()    
    
    ##  随机选择 2-ways-5-shots 10-query 数据作为测试数据 
    class_names, support_sents, query_sents = random_extract_samples(args.num_ways, args.num_shots, args.num_queries, class2sents)
    print(class_names)
    print(query_sents)
    
    support_examples = create_examples(support_sents)
    query_examples = create_examples(query_sents)
        
    support_features = convert_examples_to_features(support_examples, args.max_seq_length, tokenizer)
    query_features = convert_examples_to_features(query_examples, args.max_seq_length, tokenizer)

    support_input_ids = torch.tensor([f.input_ids for f in support_features], dtype=torch.long).to(args.device)
    support_input_mask = torch.tensor([f.input_mask for f in support_features], dtype=torch.long).to(args.device)
    support_segment_ids = torch.tensor([f.segment_ids for f in support_features], dtype=torch.long).to(args.device)

    query_input_ids = torch.tensor([f.input_ids for f in query_features], dtype=torch.long).to(args.device)
    query_input_mask = torch.tensor([f.input_mask for f in query_features], dtype=torch.long).to(args.device)
    query_segment_ids = torch.tensor([f.segment_ids for f in query_features], dtype=torch.long).to(args.device)

    # 前向计算获得 protos 和 query vectors
    support_params = (support_input_ids, support_input_mask, support_segment_ids)
    query_params = (query_input_ids, query_input_mask, query_segment_ids)
        
    loss, output = model(support_params, query_params)
    print('Loss: {:.4f} Acc: {:.4f}'.format(output['loss'], output['acc']))
        

if __name__ == '__main__':
    test()