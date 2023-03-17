#-- coding: utf-8 --
#@Date : 03/17/2023 10:01
#@Author : zxu
#@File : Constants.py
#@Software: PyCharm

d_model = 512  # 字 Embedding 的维度
d_ff = 2048  # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度
n_layers = 6  # 有多少个encoder和decoder
n_heads = 8  # Multi-Head Attention设置为8

# Encoder_input    Decoder_input        Decoder_output
sentences = [['我 是 学 生 P', 'S I am a student', 'I am a student E'],  # S: 开始符号
             ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],  # E: 结束符号
             ['我 是 男 生 P', 'S I am a boy', 'I am a boy E']]  # P: 占位符号，如果当前句子不足固定长度用P占位 pad补0

src_vocab = {'P': 0, '我': 1, '是': 2, '学': 3, '生': 4, '喜': 5, '欢': 6, '习': 7, '男': 8}  # 词源字典  字：索引

src_idx2word = {src_vocab[key]: key for key in src_vocab}
src_vocab_size = len(src_vocab)  # 字典字的个数

tgt_vocab = {'S': 0, 'E': 1, 'P': 2, 'I': 3, 'am': 4, 'a': 5, 'student': 6, 'like': 7, 'learning': 8, 'boy': 9}
idx2word = {tgt_vocab[key]: key for key in tgt_vocab}  # 把目标字典转换成 索引：字的形式
tgt_vocab_size = len(tgt_vocab)  # 目标字典尺寸

src_len = len(sentences[0][0].split(" "))  # Encoder输入的最大长度 5
tgt_len = len(sentences[0][1].split(" "))  # Decoder输入输出最大长度 5