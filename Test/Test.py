#-- coding: utf-8 --
#@Date : 03/17/2023 10:52
#@Author : zxu
#@File : Test.py
#@Software: PyCharm
import torch
import torch.utils.data as Data
from Z_NLP.Transformer.Train.Train import model
from Z_NLP.Transformer.Transformer.My_transformer import Transformer
from Z_NLP.Transformer.Utils.Constants import tgt_len, tgt_vocab, src_idx2word, idx2word, src_vocab, sentences
from Z_NLP.Transformer.Utils.DataLoader import loader

'''
tgt_len
tgt_vocab 目标字典（字：索引）
src_idx2word 源字典（索引：字）
idx2word 把目标字典转换成 索引：字的形式
src_vocab 源字典

sentences 源句子
              Encoder_input    Decoder_input        Decoder_output
sentences = [['我 是 学 生 P', 'S I am a student', 'I am a student E'],  # S: 开始符号
             ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],  # E: 结束符号
             ['我 是 男 生 P', 'S I am a boy', 'I am a boy E']]  # P: 占位符号，如果当前句子不足固定长度用P占位 pad补0
'''




# model = Transformer()

def test(model, enc_input, start_symbol):
    enc_outputs, enc_self_attns = model.Encoder(enc_input)
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model.Decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input


enc_inputs, _, _ = next(iter(loader))
predict_dec_input = test(model, enc_inputs[1].view(1, -1), start_symbol=tgt_vocab["S"])
predict, _, _, _ = model(enc_inputs[1].view(1, -1), predict_dec_input)
predict = predict.data.max(1, keepdim=True)[1]
print([src_idx2word[int(i)] for i in enc_inputs[1]], '->',
      [idx2word[n.item()] for n in predict.squeeze()])
