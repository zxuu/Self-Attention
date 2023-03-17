# transformer Decoder部分的实现  
### Decoder的完整结构
<div align=center>
<img src="https://github.com/zxuu/Self-Attention/blob/main/Pictures/Decoder.webp"/>
</div>

### Encoder-Decoder Attention的部分
  
<div align=center>
<img src="https://github.com/zxuu/Self-Attention/blob/main/Pictures/Enc-Dec-atten.webp"/>
</div>  

---
### decoder self-attention的mask由2个mask“相加”
```python
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
 ```
### Encoder-Decoder Attention 和Encoder中的Mask一样
 ```python
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
 ```
