# DLWCB

[【论文地址，官网】](http://jvs.sjtu.edu.cn/CN/abstract/abstract11911.shtml)
[【论文地址，知网】](https://doi.org/10.13465/j.cnki.jvs.2022.24.006)

## 摘要

针对滚动轴承通常在复杂条件下工作易发生故障以及训练样本较少等问题。提出一种具有全局平均池化(Global average pooling, GAP)并融合双路Laplace小波卷积和双向门控循环单元(Daul Laplace Wavelet Convolution Bidirectional Gated Recurrent Unit, DLWCB)的故障诊断方法。首先Laplace小波卷积将原始信号从时域转换为频域，接着利用双路卷积和BiGRU挖掘少量样本的多尺度和时空特征；然后设计GAP降低模型的参数量并全面融合各GRU细胞提取的时空特征。其中从优化算法和目标函数入手，引入标签平滑、AdamP等提升DLWCB应对少量样本的能力，最后实现复杂工况下故障诊断。在两种轴承数据集、有限噪声样本下，50秒内便可完成训练，达到98%以上准确率，所提方法具有良好泛化性、鲁棒性和诊断效率。
    
## 模型

![微信截图_20221219175827](https://user-images.githubusercontent.com/19371493/208401271-1d998bdd-7e84-46d7-8b29-2436bab46188.png)

## 代码
代码做了一小处重大的修改：
![image](https://user-images.githubusercontent.com/19371493/210122827-26e316fd-6dff-46e0-8e05-373118bcb298.png)

## 引用

去年录用的论文正式发表，开源了模型的代码，欢迎大家在自己的**硕/博论文**或者**期刊论文**中作为所提SOTA模型的baseline，与自己的模型相对比，可以直接替换掉自己的model作为对比。

如果有用请引用我们的论文：

```html
罗浩，何超，陈彪，路颜萍，张欣，张利. 基于Laplace小波卷积和BiGRU的少量样本故障诊断方法[J]. 振动与冲击, 2022, 41(24): 41-50.
LUO Hao，HE Chao，CHEN Biao，LU Yanping，ZHANG Xin，ZHANG Li. Small sample fault diagnosis based on Laplace wavelet convolution and BiGRU. JOURNAL OF VIBRATION AND SHOCK, 2022, 41(24): 41-50.
```

```html
@article{罗浩，何超，陈彪，路颜萍，张欣，张利,
author = {罗浩，何超，陈彪，路颜萍，张欣，张利},
title = {基于Laplace小波卷积和BiGRU的少量样本故障诊断方法},
publisher = {振动与冲击},
year = {2022},
journal = {振动与冲击},
volume = {41},
number = {24},
eid = {41},
numpages = {9},
pages = {41},
keywords = {拉普拉斯小波卷积核; 双向门控循环单元; 标签平滑；故障诊断；少量样本},
url = {http://jvs.sjtu.edu.cn/CN/abstract/article_11911.shtml},
doi = {10.13465/j.cnki.jvs.2022.24.006}
} 
```
## 特别感谢
 - https://github.com/HazeDT/WaveletKernelNet
 - 
```html
@ARTICLE{9328876,
  author={Li, Tianfu and Zhao, Zhibin and Sun, Chuang and Cheng, Li and Chen, Xuefeng and Yan, Ruqiang and Gao, Robert X.},
  journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems}, 
  title={WaveletKernelNet: An Interpretable Deep Neural Network for Industrial Intelligent Diagnosis}, 
  year={2022},
  volume={52},
  number={4},
  pages={2302-2312},
  doi={10.1109/TSMC.2020.3048950}}
} 
```
 - 
```html
@article{ZHANG2022110242,  
title = {Fault diagnosis for small samples based on attention mechanism},  
journal = {Measurement},  
volume = {187},  
pages = {110242},  
year = {2022},  
issn = {0263-2241},  
doi = {https://doi.org/10.1016/j.measurement.2021.110242 },  
url = {https://www.sciencedirect.com/science/article/pii/S0263224121011507},  
author = {Xin Zhang and Chao He and Yanping Lu and Biao Chen and Le Zhu and Li Zhang}  
} 
```

## Views
![](http://profile-counter.glitch.me/liguge/count.svg)
