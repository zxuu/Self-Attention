#-- coding: utf-8 --
#@Date : 03/17/2023 9:52
#@Author : zxu
#@File : My_transformer.py
#@Software: PyCharm


import torch.nn as nn

from Z_NLP.Transformer.Decoder.My_Decoder import Decoder
from Z_NLP.Transformer.Encoder.My_Encoder import Encoder
from Z_NLP.Transformer.Utils.Constants import d_model, tgt_vocab_size

'''
d_model 字 Embedding 的维度
tgt_vocab_size 目标字典的长度
'''
'''
# Transformer
Trasformer的整体结构，输入数据先通过Encoder，再通过Decoder，
最后把输出进行多分类，分类数为英文字典长度，也就是判断每一个字的概率。
'''
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.Encoder = Encoder()
        self.Decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):  # enc_inputs: [batch_size, src_len]
        # dec_inputs: [batch_size, tgt_len]
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)  # enc_outputs: [batch_size, src_len, d_model],
        # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.Decoder(
            dec_inputs, enc_inputs, enc_outputs)  # dec_outpus    : [batch_size, tgt_len, d_model],
        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len],
        # dec_enc_attn  : [n_layers, batch_size, tgt_len, src_len]
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = dec_logits.view(-1, dec_logits.size(-1))  # dec_logits: [-1(batch_size*tgt_len), tgt_vocab_size]
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns