# 一轮强基线上的固定消融（2026-11-23）

**状态：本地实现与CPU验证完成，服务器SSH43322拒绝连接，尚未启动GPU队列，无本轮新精度。** 用户已要求继续；采用此前24小时上限，计时从服务器实际启动开始，不能把准备时间算成GPU实验。

## 为什么这样改

完整复盘见[成果总表](results/review20261123/summary.md)。旧1epoch baseline在U1652双向R@1的两seed均值为92.3894/94.5078%，新3epoch baseline为89.6738/93.7233%（三seed）；共同seed1/2的D2S平均下降2.6813个百分点。训练地点、图片清单和评估协议已核对一致，轮数与cosine长度是明确差异，但尚不能证明过拟合或代码错误。

WT＋共识在新3epoch基线上D2S平均提高1.5744个百分点，仍未达到旧1epoch baseline。正式矩阵缺少普通共识、仅WT与普通卷积＋共识的同预算对照，无法归因。本轮回到项目原有的1epoch配方，不同时增加其他模块或调参。

## 固定七组

| ID | 特征分支 | 训练辅助目标 | 要回答的问题 |
|---|---|---|---|
| baseline | 无，原ConvNeXt-B＋GAP | 无，原双向InfoNCE | 原强配方能否恢复 |
| wtconv | WTConv | 无 | 单独WT是否有效 |
| dwconv_matched | 普通深度卷积 | 无 | 加参数本身的效果 |
| consensus | 无 | 共识 | 收益是否主要来自训练目标 |
| visible_consensus | 无 | 可见共识 | 可见权重是否比普通共识更好 |
| wt_consensus | WTConv | 共识 | WT在共识之外有无增益 |
| dw_consensus | 普通深度卷积 | 共识 | WT是否优于同参数普通卷积 |

普通分支为5层3×3深度卷积＋GELU＋逐通道增益，1024通道时与WT分支均有52,224个可训练参数；外部残差系数另有1个。WT另含32,768个固定小波参数，总参数与计算量不相同，不能称为完全相同模型。两者残差系数初值均0.1。辅助目标权重均0.5，共识/可见性保持已有轻量适配，不冒充GeoCoM/MGS²的作者实现。

所有组预先固定。负结果对照也完成正式矩阵，开发分数不删组，正式测试不选择checkpoint、不调整超参。旧16候选队列默认仍使用原来的开发集晋级规则。

## 轮数与学习率诊断

仅在训练地点中固定留20%的开发集，split seed20261122，模型seed1，U1652与SUES150m各做：

1. A：训练1轮，cosine计划1轮，即主配方的开发baseline。
2. B：训练3轮，cosine计划3轮，保存其**真实第1轮快照**并独立评估。
3. C：评估B同一次训练的第3轮权重。

B第1轮−A用于观察改变调度长度的影响；C−B第1轮用于观察同一三轮调度下继续优化的变化。它们不能单独证明过拟合，也不代表已覆盖所有可能的训练因素。诊断不用官方测试。

`train.py --checkpoint-epochs 1`会保留不可覆盖的快照与SHA；原子替换最后权重，后续epoch不会修改旧快照。`--schedule-epochs`默认等于epochs，不传参数时学习率行为不变。快照只导入参数与真实前缀训练记录，记录additional_training_steps=0；不伪造训练，也不是恢复优化器续训。

## 矩阵与预算

配置：[configs/ablation_1epoch.yaml](configs/ablation_1epoch.yaml)。物理batch24，384，单4090，其余沿用原S4G/CAMP数据及优化设置。

- 开发：7组×U1652/SUES150m×seed1/2，全部1epoch，共28次训练、56项双向评估。
- 诊断：2次三轮训练，另导入2个第1轮checkpoint，共4组产物、8项评估；开发A已包含在上一项。
- 正式：7组×5任务（U1652、SUES150/200/250/300m）×3seed，共105次1epoch训练。每个权重评双向，SUES分别评query_ids/all，共378项评估。
- 完整上限：**135次实际训练、137组产物、442项评估**。24小时截止后不开始新任务，当前任务归档后关机；提前完成则提前关机。不能保证全部矩阵在24小时内结束，不完整项不填估计值。

正式模型从ImageNet重新初始化、使用全部原训练地点；开发模型不混入正式指标。SUES仍为CAMP/DAC的120/80地点划分，query_ids与all分开比较。

## 启动与接收

服务器开机并确认当前SSH后，先部署本分支、核对数据和空间。不要重启已完成的research20261122。新驱动会在正式训练前检查7种全模型B24/384的CUDA幅值/梯度/显存及跨warmup稳定性；本地CPU检查不能代替它。

```bash
cd /root/autodl-tmp/CVGL_NEW
python scripts/research_campaign.py --config configs/ablation_1epoch.yaml --plan
nohup /root/miniconda3/bin/python scripts/research_campaign.py \
  --config configs/ablation_1epoch.yaml --hours 24 --shutdown \
  > outputs/ablation20261123_e1.launch.log 2>&1 < /dev/null &
```

本地接收沿用已有SSH复用连接，PORT/SOCKET以实际登录为准：

```bash
python scripts/collect_experiments.py --host root@connect.bjb1.seetacloud.com \
  --port PORT --socket SOCKET --run ablation20261123_e1
```

队列有单实例锁、预算边界、失败日志与续跑校验。相同配置续跑不会重复已完成任务或重置截止时间。AI修bug后必须登记repair-note并保留旧源码；改配方需另建运行名。固定对照失败会留空/标失败，不允许用别的分数组替代。

## 看什么结果

本地`outputs/ablation20261123_e1/index.html`是离线案例入口，`artifacts/<job>/D2S/report/index.html`可看失败、修好及退步；SUES all图库在`all/<direction>/report/`。每页默认100案例，完整逐图结果仍保留。

可入Git的表在`results/ablation20261123_e1/`：

- `metrics.csv`：逐seed、阶段、任务、方向、图库、实际epoch、调度长度、训练耗时与权重SHA。
- `paired_seeds.csv`：同方法三seed均值/样本标准差；n不足3时不能称完整三seed结果。
- `diagnostics.csv`：A/B/C开发集诊断与三份权重SHA。
- `ablation_effects.csv`与`ablation_summary.csv`：同seed配对的WT−普通卷积、共识−baseline、WT共识−普通共识、可见−普通共识等差值；完整性、正负seed数与标准差分列。不同图库不合并。

先看1epoch绝对分数，再看配对增益和seed是否一致，最后看修好/退步案例。不用“优于弱基线”代替达到强方法水平；三seed也不足以自动宣称显著性或机制被证明。

## 空间与留痕

本轮baseline受保护。候选仅保留正式seed1/U1652的visible_consensus与wt_consensus权重，其他权重在完整指标、queries、案例、配置、源码及SHA归档验证后释放。诊断快照完成双向评估、校验归档后释放重复文件。3GiB安全阈值保留。

启动前只回收已有完整本地备份且SHA一致的**服务器重复权重**；本地baseline与服务器唯一副本不删。准备清单在`outputs/ablation20261123_e1_preparation/remote_duplicate_plan.json`，目前尚未执行。实际清理必须再次核对服务器指纹并记录删除；不能仅凭本地清单认定已经腾出空间。代码/配方/汇总表进Git，数据、权重、outputs不进Git。
