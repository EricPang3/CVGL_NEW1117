# CVGL_NEW

将 S4G 代码整理为“训练 → 独立评估 → 查看错误 → 对照改进”的实验工具。共享 **ConvNeXt-B + GAP + 双向 InfoNCE**，支持 U1652、SUES-200 四高度的无人机—卫星双向检索，面向单张 RTX 4090。

完整命令见 [使用教程](使用教程.md)，维护契约见 [AI_README.md](AI_README.md)。

## 实验设置

| 设置 | U1652 | SUES-200 |
|---|---|---|
| 架构 / 损失 | S4G 共享编码器、GAP、InfoNCE | 相同，无 CAMP/DAC 新模块 |
| batch / epoch / seed | 24 对 / 1 / 1 | 相同 |
| 输入 / AMP | 384×384 / FP16，初始 scale=1024 | 相同 |
| 优化 | AdamW，lr=.001，weight_decay=.01，平滑=.1 | 相同 |
| 调度 | S4G cosine，warmup=.1 epoch，首步 LR=0 | 相同 |
| 增强 | S4G 原版 `s4g` | CAMP/DAC 颜色增强 `camp` |
| 划分 | 自带 train/test | 固定 120 个训练地点、80 个测试地点 |
| 主图库 | `query_ids`，只计入查询地点对应的图片 | 相同 |

S4G 原 U1652 入口 batch=128；本项目按单卡要求统一为24，不承诺原论文分数逐位复现。SUES 原始图库200地点，主评估过滤后80地点；`--gallery-scope all` 可另评完整干扰图库。SUES 配方借用 CAMP/DAC 的数据和增强设置，不加入它们的方法模块。

参考：[S4G](https://github.com/Skyy93/Sample4Geo)、[CAMP](https://github.com/Mabel0403/CAMP)，并核对同级 DAC。相关入口默认均为1epoch。新版本恢复全局Python采样RNG、NumPy同步翻转、原scheduler调用方式和AMP初始scale。保留排序后的文件清单；与原os.walk次序、GPU非确定性有关的差异仍可能存在。

## 环境和数据

AutoDL Ubuntu22.04，base Python3.10 / torch2.1.2+cu118 / torchvision0.16.2：

```bash
conda activate base
cd /root/autodl-tmp/CVGL_NEW
python -m pip install -r requirements.txt
python check_env.py
export CVGL_DATA_ROOT=/root/autodl-tmp/CVGL_NEW/data
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
```

当前服务器数据位于项目的 `data/{U1652,SUES200}`。也支持默认 `~/datasets`；路径优先级：单数据集 `--data-root` > `CVGL_DATA_ROOT/<数据集名>` > YAML/checkpoint。SUES 直接读取 `satellite-view/ID/图片`、`drone_view_512/ID/高度/图片`，无需重新复制 Training/Testing。固定划分与本地 CAMP/DAC 一致。数据、缓存、权重、全部实验输出不进入 Git。

## 训练、评估、报告

```bash
python train.py --dataset SUES200 --altitude 150 --name sues150_v2
python evaluate.py --checkpoint outputs/sues150_v2/model.pth \
  --direction D2S --output outputs/sues150_v2/D2S
python evaluate.py --checkpoint outputs/sues150_v2/model.pth \
  --direction S2D --output outputs/sues150_v2/S2D
python report.py --experiment outputs/sues150_v2/D2S
```

同一权重评双向；训练的方向参数不改变配对。训练保存最后一轮，不用测试集选 checkpoint。独立评估只需权重和测试图片，不训练、不下载预训练。

## 一次运行完整矩阵

```bash
python scripts/run_experiments.py --name baseline_v2 \
  --data-root /root/autodl-tmp/CVGL_NEW/data
```

薄脚本顺序调用三个入口：5次训练、10次评估、10份报告，失败即停止。状态、日志、指标汇总在 `outputs/baseline_v2/{run.json,run.log,metrics.csv}`。模型分别在 `outputs/baseline_v2_u1652/`、`outputs/baseline_v2_sues200_<高度>/`，每个模型有 `D2S/`、`S2D/` 评估目录。

最终下载 `outputs/baseline_v2/results.zip`，解压打开 `index.html`。包含全部案例缩略图、完整逐图JSON、配置、训练CSV和源码快照，**不含模型权重和原始分辨率图片**。默认完整图片包可能较大；`--report-limit 500` 可限量，但未打包图片不能离线查看。

`--datasets SUES200 --altitudes 150` 仅跑单高度；`--seed 2` 改种子；`--u1652-config` / `--sues200-config` 接受自定义配置。显式 `--shutdown` 只在全部成功并打包后执行 AutoDL 关机，不会自动上传结果到 GitHub。

## 本地分析和改进

本地浏览不需要GPU、数据集或Python环境。报告提供按错误/地点筛选，按排名/AP/正负分数差排序，展示最佳正负样本与Top-K；人工标注原因和笔记后导出/导入 `annotations.json`。标注不自动写回服务器。

完整报告解压后可本地对比；此时仅需 Pillow：

```bash
python -m pip install Pillow==10.4.0
python report.py --experiment outputs/local_after/sues200_150_D2S \
  --compare outputs/local_before/sues200_150_D2S
```

打开新生成的 `report/index.html` 查看指标变化、修好/退步/两次均错案例和基线Top-1。`comparison.json` 保存全部变化。程序检查查询、有效图库、预处理、任务和协议一致，训练配方允许不同。清单指纹不验证图片内容，应保持数据版本一致。

建议在训练地点内划验证集选择方案，每次只改一个因素，固定batch、seed和评估条件，再用多seed确认稳定性；正式测试用于最终报告。当前不自动划验证集，也不根据标注自动改变训练。

## 产物与指标

训练目录：`config.yaml`、`train.csv`、`model.pth`、`source.zip`；记录实际配方、环境、地点/相对路径指纹、源码hash。CSV包含实际LR、更新与跳过步数、AMP scale、采样丢弃数。

评估目录：`evaluation.yaml`、`training_config.yaml`、`metrics.json`、`queries.json`，并复制训练CSV和源码快照。保存权重SHA256、全部查询/图库清单、AP、首个正确排名、最佳正负样本、分数差。

JSON指标为0–1，终端/报告/汇总CSV为百分比。AP保留S4G梯形积分；R@1/5/10按首个正确排名。R@1%仍用工具的 `ceil(有效图库×.01)`，未复制旧代码的索引偏移；无正样本排除并计数。因此 `query_ids` 对齐候选范围，不代表所有旧指标逐位相同。

```bash
python -m unittest discover -s tests -v
```

旧 `20261117_full_seed1` 属于旧配方，保留供对照。CPU测试只验证行为和离线流程；新GPU显存、耗时和正式指标需在服务器实际运行后确认。

## 33模块实验批次

先读不可修改的[核心思想](核心思想（AI必读）.md)及[工作记录](EXPERIMENT_LOG.md)。`--module none`严格保留原模型；`--module eca`等显式开关启用共享适配层。固定配方见`configs/module_suite.yaml`，服务器由`python scripts/module_suite.py`按序续跑全部33模块和4个对照。三种Top-K方法单列，不能将适配配置称为原论文完整复现。

每次命令、源码、权重、配置与结果分别留存；本地`outputs/modules20261120/index.html`展示累积真实指标与案例链接。详见工作记录中的归档规则。

融合系数初值在训练配置，结尾值在train.csv，评估实际值在evaluation.yaml。`--module-alpha 0`关闭的是当前权重的模块旁路，不能冒充独立训练的none基线。SelfNorm统计、GeM幂与二阶矩阵运算保持FP32兼容AMP。
# 持续运行与固定种子复验

实验表保存在 `results/`：`all_metrics.csv` 为完整长表，`modules20261120_R1.csv`、`modules20261120_mAP.csv` 为首轮双向/四高度宽表，`replication.csv` 为配对seed复验。数值均为百分比，增益为百分点；未运行留空。接收程序持续更新这些表。

权重按 `configs/checkpoint_retention.json` 保留，清单见 `results/checkpoint_retention.csv`；baseline、最优候选和关键对照保留，其余只保留指标、逐图结果、报告和源码。`python scripts/experiment_records.py` 仅导表，增加 `--apply` 才清理已完成且记录完整的本地权重；手动清理前先停止接收程序，避免同时写归档清单。删除历史见 `results/weight_cleanup.csv`，实际可用权重见 `results/checkpoint_inventory.csv`。

当前批次可在服务器运行 `python scripts/run_overnight.py --until 2026-11-21T12:00:00+08:00`：等待原矩阵完成后，按原顺序使用seed2、3…补充验证，每个seed独立训练对应baseline。时间到后只停止新增补充任务，当前任务正常完成；原矩阵继续优先完整执行。

本地运行 `python scripts/collect_overnight.py --host root@服务器 --port SSH端口 --shutdown` 跟随服务器计划。复用当前SSH控制连接，不保存密码；本地电脑和网络须保持在线。总入口为 `outputs/overnight20261120/index.html`。所有已完成任务下载并校验、且预算时间已到后才请求关机；失败时保留环境。请避免同时启动原单矩阵的 `--shutdown-when-complete` 接收程序。
