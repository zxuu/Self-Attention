#-- coding: utf-8 --
#@Date : 03/17/2023 11:23
#@Author : zxu
#@File : DataLoader.py
#@Software: PyCharm


import torch
import torch.utils.data as Data
from Z_NLP.Transformer.Utils.Constants import src_vocab, tgt_vocab, sentences
'''
src_vocab = {'P': 0, '我': 1, '是': 2, '学': 3, '生': 4, '喜': 5, '欢': 6, '习': 7, '男': 8}  # 词源字典  字：索引
tgt_vocab = {'S': 0, 'E': 1, 'P': 2, 'I': 3, 'am': 4, 'a': 5, 'student': 6, 'like': 7, 'learning': 8, 'boy': 9}

               Encoder_input    Decoder_input        Decoder_output
sentences = [['我 是 学 生 P', 'S I am a student', 'I am a student E'],  # S: 开始符号
             ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],  # E: 结束符号
             ['我 是 男 生 P', 'S I am a boy', 'I am a boy E']]  # P: 占位符号，如果当前句子不足固定长度用P占位 pad补0
'''

# 把sentences 转换成字典索引
def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
      enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
      dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
      dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]
      enc_inputs.extend(enc_input)
      dec_inputs.extend(dec_input)
      dec_outputs.extend(dec_output)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)
enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
print(enc_inputs)
print(dec_inputs)
print(dec_outputs)

'''
sentences 里一共有三个训练数据，中文->英文。把Encoder_input、Decoder_input、Decoder_output转换成字典索引，
例如"学"->3、“student”->6。再把数据转换成batch大小为2的分组数据，3句话一共可以分成两组，一组2句话、一组1句话。src_len表示中文句子
固定最大长度，tgt_len 表示英文句子固定最大长度。
'''
#自定义数据集函数
class MyDataSet(Data.Dataset):
  def __init__(self, enc_inputs, dec_inputs, dec_outputs):
    super(MyDataSet, self).__init__()
    self.enc_inputs = enc_inputs
    self.dec_inputs = dec_inputs
    self.dec_outputs = dec_outputs

  def __len__(self):
    return self.enc_inputs.shape[0]

  def __getitem__(self, idx):
    return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)