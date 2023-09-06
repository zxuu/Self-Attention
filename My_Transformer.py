import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

             # Encoder_input    Decoder_input        Decoder_output
sentences = [['我 是 学 生 P' , 'S I am a student'   , 'I am a student E'],         # S: 开始符号
             ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],      # E: 结束符号
             ['我 是 男 生 P' , 'S I am a boy'       , 'I am a boy E']]             # P: 占位符号，如果当前句子不足固定长度用P占位 pad补0


src_vocab = {'P':0, '我':1, '是':2, '学':3, '生':4, '喜':5, '欢':6,'习':7,'男':8}   # 词源字典  字：索引
src_idx2word = {src_vocab[key]: key for key in src_vocab}
src_vocab_size = len(src_vocab)                 # 字典字的个数

tgt_vocab = {'S':0, 'E':1, 'P':2, 'I':3, 'am':4, 'a':5, 'student':6, 'like':7, 'learning':8, 'boy':9}
idx2word = {tgt_vocab[key]: key for key in tgt_vocab}                               # 把目标字典转换成 索引：字的形式
tgt_vocab_size = len(tgt_vocab)                                                     # 目标字典尺寸

src_len = len(sentences[0][0].split(" "))                                           # Encoder输入的最大长度 5
tgt_len = len(sentences[0][1].split(" "))                                           # Decoder输入输出最大长度 5

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

d_model = 512   # 字 Embedding 的维度
d_ff = 2048     # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度
n_layers = 6    # 有多少个encoder和decoder
n_heads = 8     # Multi-Head Attention设置为8

# 位置嵌入，position Embedding
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout=0.1,max_len=5000):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
        [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
        if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])                  # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])                  # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table)               # enc_inputs: [seq_len, d_model]
    def forward(self,enc_inputs):                                         # enc_inputs: [batch_size, seq_len, d_model]
        enc_inputs += self.pos_table[:enc_inputs.size(1),:]
        return self.dropout(enc_inputs)

'''
Mask句子中没有实际意义的占位符，例如’我 是 学 生 P’ ，P对应句子没有实际意义，所以需要被Mask，Encoder_input 和Decoder_input占位符
都需要被Mask。
这就是为了处理，句子不一样长，但是输入有需要定长，不够长的pad填充，但是计算又不需要这个pad，所以mask掉

这个函数最核心的一句代码是 seq_k.data.eq(0)，这句的作用是返回一个大小和 seq_k 一样的 tensor，只不过里面的值只有 True 和 False。如
果 seq_k 某个位置的值等于 0，那么对应位置就是 True，否则即为 False。举个例子，输入为 seq_data = [1, 2, 3, 4, 0]，
seq_data.data.eq(0) 就会返回 [False, False, False, False, True]
'''
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()# seq_q 用于升维，为了做attention，mask score矩阵用的
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0) # 判断 输入那些含有P(=0),用1标记 [seq_len, model_dim]元素全为T,F
    pad_attn_mask = pad_attn_mask.unsqueeze(1) #[batch, 1, model_dim]
    pad_attn_mask = pad_attn_mask.expand(batch_size,len_q,len_k)    # 扩展成多维度   [batch_size, len_q, len_k]
    return  pad_attn_mask

'''
# Decoder输入Mask
用来Mask未来输入信息，返回的是一个上三角矩阵。比如我们在中英文翻译时候，会先把"我是学生"整个句子输入到Encoder中，得到最后一层的输出
后，才会在Decoder输入"S I am a student"（s表示开始）,但是"S I am a student"这个句子我们不会一起输入，而是在T0时刻先输入"S"预测，
预测第一个词"I"；在下一个T1时刻，同时输入"S"和"I"到Decoder预测下一个单词"am"；然后在T2时刻把"S,I,am"同时输入到Decoder预测下一个单
词"a"，依次把整个句子输入到Decoder,预测出"I am a student E"。
'''
def get_attn_subsequence_mask(seq):                               # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]          # 生成上三角矩阵,[batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  #  [batch_size, tgt_len, tgt_len]
    return subsequence_mask

# 计算注意力信息、残差和归一化
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        :param Q: [batch_size, n_heads, len_src, d_k]
        :param K: [batch_size, n_heads, len_src, d_k]
        :param V: [batch_size, n_heads, len_src, d_v(和d_q\d_k不一定一样)]
        :param attn_mask: [batch_size, n_heads, seq_len, seq_len] attn_mask此时还是T or F
        :return:
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)   # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)                           # 如果是停用词P就等于 0 在原tensor修改
        temp  = scores
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)                                # [batch_size, n_heads, len_q, d_v]
        return context, attn

# 多头自注意力机制
# 拼接之后 输入fc层 加入残差 Norm
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)

        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):    # input_Q: [batch_size, len_q, d_model]
                                                                # input_K: [batch_size, len_k, d_model]
                                                                # input_V: [batch_size, len_v(=len_k), d_model]
                                                                # attn_mask: [batch_size, seq_len, seq_len]
        '''
        :param input_Q: enc_inputs: 词嵌入、位置嵌入之后的矩阵
        :param input_K: enc_inputs: 词嵌入、位置嵌入之后的矩阵
        :param input_V: enc_inputs: 词嵌入、位置嵌入之后的矩阵
        :param attn_mask: enc_self_attn_mask: [batch_size, src_len, src_len]元素全为T or F
        :return:
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)              # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)          # context: [batch_size, n_heads, len_q, d_v]
                                                                                 # attn: [batch_size, n_heads, len_q, len_k]
        # 拼接多头的结果
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_src, n_heads * d_v]
        output = self.fc(context)                                                # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model)(output + residual), attn


'''
## 前馈神经网络
输入inputs ，经过两个全连接层，得到的结果再加上 inputs （残差），再做LayerNorm归一化。LayerNorm归一化可以理解层是把Batch中每一句话
进行归一化。
'''
class FF(nn.Module):
    def __init__(self):
        super(FF, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))

    def forward(self, inputs):                             # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model)(output + residual)   # [batch_size, seq_len, d_model]

## encoder layer(block)
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()                                     # 多头注意力机制
        self.pos_ffn = FF()                                        # 前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):                                # enc_inputs: [batch_size, src_len, d_model]
        '''
        :param enc_inputs: 词嵌入、位置嵌入之后的矩阵
        :param enc_self_attn_mask: [batch_size, src_len, src_len]元素全为T or F
        :return:
        '''
        #输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V                             # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,    # enc_outputs: [batch_size, src_len, d_model],
                                               enc_self_attn_mask)                    # attn: [batch_size, n_heads, src_len, src_len]
        # 多头自注意力机制之后（Add & Norm之后），进行FF(Add & Norm)
        enc_outputs = self.pos_ffn(enc_outputs)                                       # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

'''
## Encoder
第一步，中文字索引进行Embedding，转换成512维度的字向量。
第二步，在子向量上面加上位置信息。
第三步，Mask掉句子中的占位符号。
第四步，通过6层的encoder（上一层的输出作为下一层的输入）。
'''
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer() for _ in range(n_layers)]
        )

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns




# 测试
'''
enc_inputs:
tensor([[1, 2, 3, 4, 0],
        [1, 5, 6, 3, 7],
        [1, 2, 8, 4, 0]])
'''
#enc_outputs, enc_self_attns = Encoder()(enc_inputs)
#print(enc_outputs.shape)    # torch.Size([3, 5, 512])





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


'''
# 定义网络
'''
model = Transformer()
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 占位符 索引为0.
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

'''
# 训练Transformer
因为batch=2，所以一个epoch有两个loss
'''
for epoch in range(50):
    for enc_inputs, dec_inputs, dec_outputs in loader:
        enc_inputs, dec_inputs, dec_outputs = enc_inputs, dec_inputs, dec_outputs
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        # dec_outputs: tensor([[3, 4, 5, 6, 1], [3, 7, 8, 2, 1]])
        temp = dec_outputs.view(-1)
        loss = criterion(outputs, dec_outputs.view(-1))  # outputs: [-1(batch_size*tgt_len), tgt_vocab_size]
        # dec_outputs.view(-1): tensor([3, 4, 5, 6, 1, 3, 7, 8, 2, 1])
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# 测试

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