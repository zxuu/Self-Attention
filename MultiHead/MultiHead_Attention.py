#-- coding: utf-8 --
#@Date : 03/01/2023 17:35
#@Author : zxu
#@File : MultiHead_Attention.py
#@Software: PyCharm

import torch
import torch.nn.functional as F
import math
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self):
        super(MultiHeadAttention, self).__init__()

    def self_attention(self, query, key, value, dropout=None, mask=None):
        d_k = query.size(-1)  # d_k=单词的embedding长度
        # K矩阵转置一下：Q*K^T
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        #print('scores:', scores)

        # mask的操作在QK之后，softmax之前
        if mask is not None:
            mask.cuda()
            scores = scores.masked_fill(mask == 0, -1e9)
            print('scores_masked:', scores)

        self_attn = F.softmax(scores, dim=-1)
        #print('self_attn_softmax:', scores)

        if dropout is not None:
            self_attn = dropout(self_attn)
            print('self_attn_dropout:', scores)

        return torch.matmul(self_attn, value), self_attn

    def forward(self,  head, d_model, query, key, value, dropout=0.1,mask=None):
        """

        :param head: 头数，8
        :param d_model: 输入的维度(Q,K的维度) 16
        :param query: Q
        :param key: K
        :param value: V
        :param dropout: 暂时用不到
        :param mask: 暂时用不到
        :return: 多头拼接之后在经过线性变换的结果
        """
        assert (d_model % head == 0)
        self.d_k = d_model // head    # d_k:2
        self.head = head
        self.d_model = d_model

        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)

        # 自注意力机制的 QKV 同源，线性变换

        # 最后输出得到Z的时候也要做一次线性变换
        self.linear_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.attn = None



        # if mask is not None:
        #     # 多头注意力机制的线性变换层是4维，是把query[batch, frame_num, d_model]变成[batch, -1, head, d_k]
        #     # 再1，2维交换变成[batch, head, -1, d_k], 所以mask要在第一维添加一维，与后面的self attention计算维度一样
        #     mask = mask.unsqueeze(1)

        n_batch = query.size(0)

        # 多头需要对这个 X 切分成多头
        #Q[b,4,16],K[b,4,16],V[b,4,16]
        temp = self.linear_query(query)    # [b,4,16]
        # 16截断为[8,2]  -1：这一维度自适应
        temp2 = temp.view(n_batch, -1, self.head, self.d_k)    # [b,4,8,2]
        query = temp2.transpose(1, 2)  # [b, 8, 4, 2]
        #print(query.shape)
        key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 4, 2]
        #print(key.shape)
        value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 4, 2]
        #print(value.shape)


        x, self.attn = self.self_attention(query, key, value, mask=mask)    # [b,8,4,2]
        # 变为三维， 或者说是concat head(拼接各个head-attention的结果)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)    # [b,4,16]
        x = self.linear_out(x)    # [b,4,16]
        print(x.shape)
        return x

# Inputs to the attention module
batch_size = 3    # 每次选取3句话
dim_embedding = 16    #input_size(embedding维度(正常来说是512，为了方便设置为16)。后面8个头的话，要进行截断。2个维度一份，一共8份)
sequence_length = 4    #每句话固定有4个单词(单词之间计算注意力)
head_num = 8    # 8个头
#dim_V = 8    #V向量的长度(V向量长度可以与Q and K不一样)
#dim_QK = 7    #Q and K向量的长度(dim_embedding经过Wq、Wk变换后QK向量长度变为dim_QK)

query = torch.randn(batch_size, sequence_length, dim_embedding)
keys = torch.randn(batch_size, sequence_length, dim_embedding)
values = torch.randn(batch_size, sequence_length, dim_embedding)

MultiHeadAttention = MultiHeadAttention()
att = MultiHeadAttention(head=head_num, d_model=dim_embedding, query=query, key=keys, value=values, dropout=0.1,mask=None)
#print(att)
