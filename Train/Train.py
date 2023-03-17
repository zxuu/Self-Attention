#-- coding: utf-8 --
#@Date : 03/17/2023 10:10
#@Author : zxu
#@File : Train.py
#@Software: PyCharm

import torch.nn as nn
import torch.optim as optim
from Z_NLP.Transformer.Transformer.My_transformer import Transformer
from Z_NLP.Transformer.Utils.DataLoader import loader


# 定义网络
model = Transformer()
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 占位符 索引为0.
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

'''
# 训练Transformer
因为batch=2，所以一个epoch有两个loss
'''
for epoch in range(2):
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