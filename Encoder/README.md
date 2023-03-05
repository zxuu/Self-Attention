# Transformer中的完整的Encoder模块（只是没有dropout操作）  
<div align=center>
<img src="[https://raw.githubusercontent.com/zxuu/Self-Attention/main/Pictures/Encoder.webp"/>
</div>   

### 位置嵌入  
<div align=center>
<img src="https://raw.githubusercontent.com/zxuu/Self-Attention/main/Pictures/position_mask.webp"/>
</div>  

### Q、K、V相乘（scaled Dot-Product Attention）
<div align=center>
<img src="https://raw.githubusercontent.com/zxuu/Self-Attention/main/Pictures/scaled_Dot-produnct-Attention.webp"/>
</div>  

### encoder_layer(block)  本例设置6个encoder
<div align=center>
<img src="https://raw.githubusercontent.com/zxuu/Self-Attention/main/Pictures/encoder_layer(block).webp"/>
</div>  

### ff-add&norm 前馈网络->残差连接->norm归一化
<div align=center>
<img src="https://raw.githubusercontent.com/zxuu/Self-Attention/main/Pictures/ff-add%26norm.webp"/>
</div>  


# 关于掩码Mask详见该博客：[自注意力中的不同的掩码介绍以及他们是如何工作的？](https://baijiahao.baidu.com/s?id=1746456529779226584&wfr=spider&for=pc)  
![](https://github.com/zxuu/Self-Attention/blob/main/Pictures/Encoder_Mask.png)
