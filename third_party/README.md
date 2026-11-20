# 组件来源与适配

`SOURCES.json`记录实际下载URL、分支、上游源码SHA256和许可证快照；各组件目录保留LICENSE及原始`upstream.py.txt`。来源是用户模块目录中的作者仓库，不修改同级参考项目。

多数`components/*/layer.py`仅从原文件提取所需类/函数并移除无关主干和框架import。以下变化必须与“原论文完整复现”区分：

- GCBlock/CAA的mmcv初始化和Conv/BN/SiLU接口由`compat.py`提供，CAA使用PyTorch原生初始化；不安装整个mmcv/mmengine。
- WTConv保留作者实现，wavelet helper仅支持本轮注册的db1/Haar闭式滤波器，省去PyWavelets；不声称支持其他小波。
- GRN转BHWC再转回；PAM经128通道投影，并保留其零增量初始化；iSQRT-COV显式FP32，展平作者返回的尾部单维。
- SelfNorm统计与GeM幂运算显式FP32，避免FP16下方差eps消失或幂溢出；新模块的空间输出再按原特征dtype送回原head。
- ACB、RepVGG、SRE使用1024→128→1024适配瓶颈；UniRepLK使用普通PyTorch卷积回退。缩小参数在实际配置、代码与参数计数中留痕。
- BoQ投影128、8queries、1layer、row_dim2；SALAD为8clusters/32cluster_dim/128token_dim；EDT沿用作者SA_CA并缩小查询/解码维度。所有并行分支保留原GAP。
- SE直接复用安装的torchvision 0.16.2 SqueezeExcitation（BSD-3-Clause），不复制另一套实现。
- burst_gap是独立的固定规则思想适配，未复制VLAD-BuFF的完整VLAD/预投影管线，不应标成VLAD-BuFF原方法复现。
- Patch-NetVLAD保留作者多尺度局部描述子，以训练地点PCA/KMeans适配ConvNeXt；匹配使用互近邻patch余弦，不包括原VGG/RANSAC管线。
- R²Former保留作者成对头，将硬编码CUDA改为输入device，使用CNN范数显著性/网格坐标/训练地点PCA描述子；正负对只来自训练地点。
- SuperGlobal保留作者MDescAug/RerankwMDA，改成输入device复制，候选预算统一20，近邻来自当前查询的参考候选，不使用其他测试查询。

这些移植实验评估的是在当前CVGL基线上的效果；原领域论文结果不能作为当前任务的预期增益。
