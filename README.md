# Self-Attention（pytorch实现）
transformer开山之作：[Attention is all you need](https://arxiv.org/abs/1706.03762)
代码参考这篇博客[手撕Transformer Transformer输入输出细节以及代码实现（pytorch）](https://blog.51cto.com/u_14300986/5467368)  
<div align=center>
<img src="./Pictures/transformer-architecture.png" width="300"/>
</div>

1. decoder中，查询来自Decoder，键和值来自Encoder的输出。encoder的输出作为decoder的K,V的来源
2. 解码器一个Block包含两个多头自注意力机制：
   - Masked Multi-Head Attention：decoder输入文本每个token只能观察到前面的token，因此需要mask。
   - Multi-Head Attention：decoder输出文本每个token可以观察到encoder的所有token，因此不需要mask。

---

**OneHead文件夹**下是单头自注意力机制的实现(无Mask)  
**MultiHead文件夹**下是多头自注意力机制的实现(无Mask)  
**Encoder文件夹**下是Ecoder编码器的完整实现（有mask）  
**Decoder文件夹**下是Decoder解码器的完整实现（有mask）  
**Transformer文件夹**下是把Encoder和Decoder两部分的合并，形成完整的transformer结构  

**Train文件夹**是训练部分  
**Test文件夹**是测试部分  
**Utils文件夹**是工具包，包括一些常量的定义，还有Dataloader  

**My_Transformer.py**是完整的transformer构建、数据输入、测试的例子  
# torch、transformer版本
```bash
torch                          1.13.1+cu117
torchaudio                     0.13.1+cu117
torchvision                    0.14.1+cu117
transformers                   4.26.1
```
# 输入例子
```python
             # Encoder_input    Decoder_input        Decoder_output
sentences = [['我 是 学 生 P' , 'S I am a student'   , 'I am a student E'],         # S: 开始符号
             ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],      # E: 结束符号
             ['我 是 男 生 P' , 'S I am a boy'       , 'I am a boy E']]             # P: 占位符号，如果当前句子不足固定长度用P占位 pad补0
```

# My_transformer.py输出(直接运行即可)
```bash
......
Epoch: 0050 loss = 0.088758
Epoch: 0050 loss = 0.003786
['我', '是', '学', '生', 'P'] -> ['I', 'am', 'a', 'student', 'E']
```
