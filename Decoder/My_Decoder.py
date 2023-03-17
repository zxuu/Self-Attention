#-- coding: utf-8 --
#@Date : 03/15/2023 9:39
#@Author : zxu
#@File : My_Decoder.py
#@Software: PyCharm


from Z_NLP.Transformer.Encoder.My_Encoder import MultiHeadAttention, FF, PositionalEncoding, get_attn_pad_mask, \
    get_attn_subsequence_mask, Encoder

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from Z_NLP.Transformer.Utils.Constants import tgt_vocab_size,d_model,n_layers


'''
    d_model = 512  # 字 Embedding 的维度
    n_layers = 6  # 有多少个encoder和decoder
    n_heads = 8  # Multi-Head Attention设置为8
'''

# decoder layer(block)
# decoder两次调用MultiHeadAttention时，第一次调用传入的 Q，K，V 的值是相同的，都等于dec_inputs，第二次调用 Q 矩阵是来自Decoder的
# 输入。K，V 两个矩阵是来自Encoder的输出，等于enc_outputs。
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = FF()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask,
                dec_enc_attn_mask):  # dec_inputs: [batch_size, tgt_len, d_model]
        # enc_outputs: [batch_size, src_len, d_model]
        # dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        # dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        # decoder的self-attention
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs,
                                                        dec_self_attn_mask)  # dec_outputs: [batch_size, tgt_len, d_model]
        # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        # decoder自注意力之后的值作为Q值。K,V来自Encoder的输出
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                      enc_outputs,
                                                      dec_enc_attn_mask)  # dec_outputs: [batch_size, tgt_len, d_model]
        # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs = self.pos_ffn(dec_outputs)  # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


'''
# Decoder
第一步，英文字索引进行Embedding，转换成512维度的字向量。
第二步，在子向量上面加上位置信息。
第三步，Mask掉句子中的占位符号和输出顺序.
第四步，通过6层的decoder（上一层的输出作为下一层的输入）
'''


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        enc_intpus: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        enc_outputs: [batch_size, src_len, d_model]
        '''
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, tgt_len, d_model]
        # Decoder输入序列的pad mask矩阵（这个例子中decoder是没有加pad的，实际应用中都是有pad填充的）
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)  # [batch_size, tgt_len, tgt_len]
        ''' 'S I like learning P'  'S I am a student'
        tensor([[[ True, False, False, False, False],
                 [ True, False, False, False, False],
                 [ True, False, False, False, False],
                 [ True, False, False, False, False],
                 [ True, False, False, False, False]],

                [[ True, False, False, False, False],
                 [ True, False, False, False, False],
                 [ True, False, False, False, False],
                 [ True, False, False, False, False],
                 [ True, False, False, False, False]]])'''
        # Masked Self_Attention：当前时刻是看不到未来的信息的
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(
            dec_inputs)  # [batch_size, tgt_len, tgt_len] 下三角包括对角线为0，上三角为1
        '''
        tensor([[[0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1],
                 [0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0]],

                 [0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1],
                 [0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0]]], dtype=torch.uint8)'''
        # Decoder中把两种mask矩阵相加（既屏蔽了pad的信息，也屏蔽了未来时刻的信息）
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0)  # [batch_size, tgt_len, tgt_len]
        '''tensor([[[ True,  True,  True,  True,  True],
                    [ True, False,  True,  True,  True],
                    [ True, False, False,  True,  True],
                    [ True, False, False, False,  True],
                    [ True, False, False, False, False]],

                    [ True,  True,  True,  True,  True],
                    [ True, False,  True,  True,  True],
                    [ True, False, False,  True,  True],
                    [ True, False, False, False,  True],
                    [ True, False, False, False, False]]])'''
        # 这个mask主要用于encoder-decoder attention层
        # get_attn_pad_mask主要是enc_inputs的pad mask矩阵(因为enc是处理K,V的，求Attention时是用v1,v2,..vm去加权的，
        # 要把pad对应的v_i的相关系数设为0，这样注意力就不会关注pad向量)
        #                       dec_inputs只是提供expand的size的
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]
        '''tensor([[[False, False, False, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False]],

                    [False, False, False, False,  True],
                    [False, False, False, False,  True],
                    [False, False, False, False,  True],
                    [False, False, False, False,  True],
                    [False, False, False, False,  True]]])'''

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns