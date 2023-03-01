from math import sqrt

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, dim_embedding, dim_qk, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_embedding = dim_embedding
        self.dim_qk = dim_qk
        self.dim_v = dim_v

        # 定义线性变换函数
        self.linear_q = nn.Linear(dim_embedding, dim_qk, bias=False)
        self.linear_k = nn.Linear(dim_embedding, dim_qk, bias=False)
        self.linear_v = nn.Linear(dim_embedding, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_embedding)

    def forward(self, x):
        # x: batch, sequence_length, dim_embedding
        # 根据文本获得相应的维度

        batch, n, dim_embedding = x.shape
        assert dim_embedding == self.dim_embedding

        q = self.linear_q(x)  # batch, sequence_length, dim_embedding
        print(q.shape)
        k = self.linear_k(x)  # batch, sequence_length, dim_embedding
        print(k.shape)
        v = self.linear_v(x)  # batch, sequence_length, dim_embedding
        print(v.shape)
        # q*k的转置 并*开根号后的dk
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, sequence_length, sequence_length
        # 归一化获得attention的相关系数
        dist = torch.softmax(dist, dim=-1)  # batch, sequence_length, sequence_length
        # attention注意力分数和v相乘，获得最终的得分
        att = torch.bmm(dist, v)
        return att

# Inputs to the attention module
batch_size = 3    # 每次选取3句话
dim_embedding = 6    #input_size
sequence_length = 4    #每句话固定有4个单词(单词之间计算注意力)
dim_V = 8    #V向量的长度(V向量长度可以与Q and K不一样)
dim_QK = 7    #Q and K向量的长度(dim_embedding经过Wq、Wk变换后QK向量长度变为dim_QK)

# 输入的数据
x = [[[ 0.1532, -0.0044,  0.6197,  0.6332, -1.1813,  0.3393],
         [ 0.4235, -0.1569,  0.1241,  0.0882,  0.2117,  1.0789],
         [-0.1987, -0.2671, -0.2119, -0.3006, -0.1815,  0.0309],
         [ 0.5800, -0.0540, -0.3753, -0.3589,  0.0209,  0.0198]],

        [[ 0.6163, -0.1154, -0.4507,  0.0220, -0.2439, -0.2730],
         [-0.2422, -0.2329,  0.4126, -0.1035, -0.7506,  0.6158],
         [-1.1735,  0.1261,  0.1183, -0.5666, -0.0784,  1.1600],
         [ 1.4221,  0.5371,  1.0822,  0.3618, -1.1405, -0.2908]],

        [[-1.4205, -0.3030,  0.6546,  0.1172,  0.0054,  0.2997],
         [ 0.1374,  0.1845,  0.0117, -0.2321, -0.5516, -0.0238],
         [-0.2277,  0.2537, -0.0104, -0.1850,  0.3958,  0.7365],
         [ 0.4063, -0.1534,  0.2482,  0.2756,  0.7651,  1.8565]]]
x = torch.tensor(x)

x_gen = torch.randn(batch_size, sequence_length, dim_embedding)
attention = SelfAttention(dim_embedding, dim_QK, dim_V)
att = attention(x_gen)
print(att)