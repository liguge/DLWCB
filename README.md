# DLWCB

[【论文地址，官网】](http://jvs.sjtu.edu.cn/CN/abstract/abstract11911.shtml)
[【论文地址，知网】](https://doi.org/10.13465/j.cnki.jvs.2022.24.006)

## 摘要

针对滚动轴承通常在复杂条件下工作易发生故障以及训练样本较少等问题。提出一种具有全局平均池化(Global average pooling, GAP)并融合双路Laplace小波卷积和双向门控循环单元(Daul Laplace Wavelet Convolution Bidirectional Gated Recurrent Unit, DLWCB)的故障诊断方法。首先Laplace小波卷积将原始信号从时域转换为频域，接着利用双路卷积和BiGRU挖掘少量样本的多尺度和时空特征；然后设计GAP降低模型的参数量并全面融合各GRU细胞提取的时空特征。其中从优化算法和目标函数入手，引入标签平滑、AdamP等提升DLWCB应对少量样本的能力，最后实现复杂工况下故障诊断。在两种轴承数据集、有限噪声样本下，50秒内便可完成训练，达到98%以上准确率，所提方法具有良好泛化性、鲁棒性和诊断效率。
    
## 模型

![微信截图_20221219175827](https://user-images.githubusercontent.com/19371493/208401271-1d998bdd-7e84-46d7-8b29-2436bab46188.png)

## 代码
代码做了一小处重大的修改：
![91480cbed8fd4e2dbc574d355949857](https://user-images.githubusercontent.com/19371493/210123166-e2284bfa-c346-4e27-8c22-40cafdc6b991.png)

## 引用

去年录用的论文正式发表，开源了模型的代码，欢迎大家在自己的**硕/博论文**或者**期刊论文**中作为所提SOTA模型的baseline，与自己的模型相对比，可以直接替换掉自己的model作为对比。

如果有用请引用我们的论文：

```html
He, Chao, Hongmei Shi, Jin Si, and Jianbo Li. "Physics-informed interpretable wavelet weight initialization and balanced dynamic adaptive threshold for intelligent fault diagnosis of rolling bearings." Journal of Manufacturing Systems 70 (2023): 579-592.
He, Chao, Hongmei Shi, Xiaorong Liu, and Jianbo Li. "Interpretable physics-informed domain adaptation paradigm for cross-machine transfer diagnosis." Knowledge-Based Systems 288 (2024): 111499.
```

```html
@article{he2023physics,
  title={Physics-informed interpretable wavelet weight initialization and balanced dynamic adaptive threshold for intelligent fault diagnosis of rolling bearings},
  author={He, Chao and Shi, Hongmei and Si, Jin and Li, Jianbo},
  journal={Journal of Manufacturing Systems},
  volume={70},
  pages={579--592},
  year={2023},
  publisher={Elsevier}
}
@article{he2024interpretable,
  title={Interpretable physics-informed domain adaptation paradigm for cross-machine transfer diagnosis},
  author={He, Chao and Shi, Hongmei and Liu, Xiaorong and Li, Jianbo},
  journal={Knowledge-Based Systems},
  volume={288},
  pages={111499},
  year={2024},
  publisher={Elsevier}
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
# 环境

pytorch == 1.10.0  
python ==  3.8  
cuda ==  10.2   

# 联系
- Chao He
- chaohe#bjtu.edu.cn   (please replace # by @)

## Views
![](http://profile-counter.glitch.me/liguge/count.svg)
