# coding=utf-8

import torch

def euclidean_dist(q_vecs, protos):
    """
    计算查询样本向量 q_vecs 和原型向量 protos 之间的欧式距离
    
    Args:
        q_vecs: shape of [num_ways*num_queries, dim]
        protos: shape of [num_ways, dim]
    """
    n = q_vecs.size(0)
    num_ways = protos.size(0)
    dim = q_vecs.size(1)
    assert dim == protos.size(1)

    x = q_vecs.unsqueeze(1).expand(n, num_ways, dim)
    y = protos.unsqueeze(0).expand(n, num_ways, dim)

    return torch.pow(x - y, 2).sum(2)