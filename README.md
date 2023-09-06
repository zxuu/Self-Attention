# Self-Attention（pytorch实现）
参考这篇博客[手撕Transformer Transformer输入输出细节以及代码实现（pytorch）](https://blog.51cto.com/u_14300986/5467368)  
  
**OneHead文件夹**下是单头自注意力机制的实现(无Mask)  
**MultiHead文件夹**下是多头自注意力机制的实现(无Mask)  
**Encoder文件夹**下是Ecoder编码器的完整实现（有mask）  
**Decoder文件夹**下是Decoder解码器的完整实现（有mask）  
**Transformer文件夹**下是把Encoder和Decoder两部分的合并，形成完整的transformer结构  
  
---  
**Train文件夹**是训练部分  
**Test文件夹**是测试部分  
**Utils文件夹**是工具包，包括一些常量的定义，还有Dataloader  
  
---  
**My_Transformer.py**是完整的transformer构建、数据输入、测试的例子  

---  
**输入例子**  
```python
             # Encoder_input    Decoder_input        Decoder_output
sentences = [['我 是 学 生 P' , 'S I am a student'   , 'I am a student E'],         # S: 开始符号
             ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],      # E: 结束符号
             ['我 是 男 生 P' , 'S I am a boy'       , 'I am a boy E']]             # P: 占位符号，如果当前句子不足固定长度用P占位 pad补0
```
