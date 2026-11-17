# CVGL_NEW

基于 CVGL_EP / Sample4Geo (S4G) 的精简实验工具：一个共享 ConvNeXt-B + GAP 编码器，使用 batch 内双向 InfoNCE，支持 U1652、SUES-200 的无人机→卫星 (D2S) 和卫星→无人机 (S2D) 检索。

保留原项目的编码器、可学习温度、标签平滑、同步水平翻转、两路增强和 batch 内地点不重复约束。没有额外投影头、分类器、多卡逻辑或新的挖掘模块。训练和评估独立：同一份权重可以评两个方向，无需分别训练。这里保留的是 S4G 架构基线，并非承诺复现论文全部采样策略或公布分数。

## 环境

目标服务器：AutoDL、Ubuntu 22.04、单张 RTX 4090。直接复用镜像的 **base 环境（Python 3.10、PyTorch 2.1.2、CUDA 11.8）**；torchvision 对应 0.16.2。无需创建新环境或重装 torch。

```bash
conda activate base
cd /root/autodl-tmp
git clone https://github.com/EricPang3/CVGL_NEW1117.git CVGL_NEW
cd CVGL_NEW

python -m pip install -r requirements.txt
python check_env.py
```

`requirements.txt` 只列项目依赖，引用 `constraints-autodl.txt` 约束 torch/torchvision，防止间接依赖升级框架。`check_env.py` 检查版本和可见 GPU，并执行一次 torchvision CUDA 算子；检查过程不下载权重、不训练模型。如镜像缺少 torchvision 或版本不匹配，参见 [使用教程](使用教程.md#11-安装环境) 的单独修复命令。

默认 384×384、24 对图片/batch、FP16 AMP、单卡 `cuda:0`；每个 batch 实际输入 48 张图片。评估 batch 为 64。如遇显存不足，可降低 `--batch-size`。CPU 仅用于小规模功能检查，不建议正式训练。

首次训练默认通过 timm 下载 ImageNet 预训练参数。将缓存放到 AutoDL 数据盘：

```bash
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE"
```

已有缓存包的导入方法见使用教程。每次新开终端需重新设置变量。首次成功下载后可以复用缓存，缓存不进入 Git。

独立评估始终 `pretrained=False`，只加载指定权重。`--no-pretrained` 可用于随机初始化功能测试，正式基线使用预训练初始化。

## 数据

本地默认只读 `~/datasets`；AutoDL 将图片单独上传到 `/root/autodl-tmp/datasets`，然后设置：

```bash
export CVGL_DATA_ROOT=/root/autodl-tmp/datasets
```

路径优先级为 **`--data-root` > `CVGL_DATA_ROOT/<数据集名>` > YAML/权重配置中的路径**。环境变量是两个数据集的父目录；`--data-root` 是某一个数据集目录。这样本地与服务器共享同一份配置，无需改动 YAML。程序不复制、不重排源文件，图片与地点按路径排序，非图片文件会被忽略。

```text
/root/autodl-tmp/datasets/             # 本地可放 ~/datasets/
├── U1652/
│   ├── train/{satellite,drone}/<地点>/<图片>
│   └── test/{query_drone,gallery_satellite,query_satellite,gallery_drone}/<地点>/<图片>
└── SUES200/
    ├── satellite-view/<0001…0200>/<图片>
    └── drone_view_512/<0001…0200>/<150|200|250|300>/<图片>
```

SUES 直接读取原始目录；`configs/sues200_split.yaml` 沿用 CVGL_EP 的 120 个训练地点，其余 80 个地点作查询，图库包含全部 200 个地点。每次评估导出完整图库清单以及实际采用的范围。

U1652 训练配对取每个地点排序后的第一张卫星图，对应该地点所有无人机图，与旧代码的一张卫星对多张无人机约定一致。训练时缺少匹配视角会报错。评估只需要 test 目录，SUES 只读取测试查询和图库所需图片。

## 使用

在项目根目录执行。`--dataset` 自动选择对应 YAML；也可以用 `--config configs/sues200.yaml` 指定配置。命令行覆盖 YAML 中的同名项。

```bash
# U1652 训练
python train.py --dataset U1652 --name u1652_baseline

# SUES 训练：方向仅记录默认评估任务，不改变训练配对或损失
python train.py --dataset SUES200 --altitude 150 --direction D2S --name sues150

# 使用配置文件、修改训练参数
python train.py --config configs/sues200.yaml --altitude 300 --epochs 5 --batch-size 24 --name sues300

# 独立评估：从新权重内自动恢复配置
python evaluate.py --checkpoint outputs/sues150/model.pth

# 同一权重评另一方向：使用独立目录保存结果
python evaluate.py --checkpoint outputs/sues150/model.pth --direction S2D --output outputs/sues150_s2d

# 评已有 CVGL_EP 权重：显式指定数据集；支持旧 model.* + logit_scale state_dict
python evaluate.py --checkpoint /path/to/old_weights.pth --dataset U1652 --direction D2S --output outputs/old_u1652

# 用同一模型评其他高度
python evaluate.py --checkpoint outputs/sues150/model.pth --dataset SUES200 --altitude 300 --output outputs/sues150_test300

# 报告默认收录排名最差的 200 个案例；0 表示全部
python report.py --experiment outputs/sues150
python report.py --experiment outputs/sues150 --limit 0
```

训练默认保存最后一轮权重，每轮原子替换 `model.pth`，训练曲线完整保存在 CSV；没有根据测试集挑选最佳模型。`--init-checkpoint` 可加载初始权重，但不恢复优化器。独立评估不使用优化器，也不执行训练。已有评估文件默认拒绝覆盖；需要重新计算时用新 `--output` 或明确传入 `--overwrite`。

新 checkpoint 包含模型参数、实际配置和 epoch。旧 checkpoint 没有配置时按指定 YAML 解释，输入尺寸默认 384；如旧模型以其他尺寸训练，需要显式传 `--img-size`。仅支持当前 ConvNeXt-B 结构，参数不匹配会报错。

## 输出

```text
outputs/<实验名>/
├── config.yaml       # 训练配置、训练地点、模型与环境信息
├── train.csv         # 每轮 loss、lr、耗时、有效/丢弃样本数、steps、logit_scale
├── model.pth         # 最后一轮参数 + 配置 + epoch
├── evaluation.yaml   # 本次评估实际参数、权重路径及 SHA256
├── metrics.json      # 总体指标、查询/图库数量、评估口径
├── queries.json      # 全部逐图结果、图库清单、评估参数
├── report.zip        # 解压后直接双击 index.html
└── report/
    ├── index.html
    └── images/       # 已复制的缩略图，不依赖原数据路径
```

评估输出到新目录时，`config.yaml` 记录本次配置；原训练目录内的配置不会被评估覆盖，因此另存 `evaluation.yaml`。逐图结果含原图路径、原始地点 ID、首个正确排名（从 1 开始）、AP、正样本数、Top-K 候选与相似度、最佳正确图片；S2D 的多个正确候选均参与 AP。完整图库清单包括被排除项，报告只将选中案例的缩略图打包。

报告支持正确/错误/无匹配筛选、地点或文件名搜索、分页。JSON 直接嵌入页面，无 CDN、fetch 或服务器依赖。下载 `report.zip` 即可离线查看；重建报告仍需要访问源图片。`--limit` 仅限制报告收录数量，不影响评估结果。

## 指标口径

- `metrics.json` 和逐图 AP 使用 **0–1**，终端和报告显示百分比。
- L2 归一化后，以 FP32 点积计算余弦相似度。GPU AMP 仅用于编码器，归一化在 FP32 中执行。
- 正确匹配定义为地点 ID 相同；同一 query 可以有多个正确 gallery。
- 默认 `gallery_scope: all`，保留完整图库中的干扰地点。`--gallery-scope query_ids` 沿用旧项目按 query 地点集合过滤 gallery 的规则；过滤数量写入结果。
- 同分按排序后的文件路径稳定排序；无正确匹配的查询保留记录，AP/排名为 null，从指标分母中排除并单独计数；全部无匹配则报错。
- AP 沿用旧 S4G `compute_mAP` 的梯形积分，并非 sklearn 的阶梯式 average precision。R@K 是首个正确候选是否在前 K；R@1% 使用有效图库数量的 `ceil(0.01 × G)`，至少为 1。
- 新工具修正了旧 R@1% 的索引与分母歧义。`query_ids` 只兼容图库过滤，**不表示全部旧指标逐位一致**；默认完整图库的分数也不能直接与旧过滤口径比较。
- 原训练元数据会随 checkpoint 保留，跨高度或跨数据集评估时另记录测试配置。可核对训练地点、评估清单与权重哈希；未对所有图像内容计算哈希。

## 代码与检查

`train.py` 组织训练，`evaluate.py` 组织评估，`report.py` 生成离线报告。`sample4geo/dataset` 负责数据与原增强，`trainer.py` 执行单轮优化，`sample4geo/evaluate` 提取特征、排名和计算指标。核心类继续沿用 U1652 命名供两数据集共用。

```bash
python -m unittest discover -s tests -v
```

检查覆盖数据路径优先级、多正样本 AP、R@1% 边界、无正确匹配、同分排序、地点唯一采样、InfoNCE 反传、原始 SUES 划分和离线报告打包。CPU 功能检查不代表 4090 显存、性能或论文指标复现通过。

初次重构验证：6 项单元检查通过；真实 ConvNeXt-B 在 32×32 合成图片上完成一次 CPU 更新，保存后在只含测试目录的数据根下完成双向评估、旧 state_dict 加载及报告打包；同权重的旧/新编码器输出和 InfoNCE 数值对齐。报告脚本通过 DOM 模拟的渲染、筛选和搜索检查，未做真实浏览器渲染。真实数据只做路径与划分扫描：U1652 训练 701 个地点，D2S 查询/图库 37855/951 张、S2D 为 701/51355 张；SUES 每高度 D2S 为 4000/200 张、S2D 为 80/10000 张。未在真实数据上启动完整训练或计算正式指标。

本轮 AutoDL 适配后，共 7 项测试通过；依赖约束兼容 `+cu118`，数据/权重/实验输出的 Git 忽略规则已检查。开发机只有 CPU，`check_env.py` 会正确报告 GPU 环境不满足；尚未在用户服务器上运行该检查或开展正式 GPU 训练。

代码改写自同级 CVGL_EP 的 Sample4Geo 基线；SUES 划分直接沿用该项目提供的列表。保留来源说明，具体论文复现实验仍需核对相同预训练、数据划分、采样和评估协议。

## GitHub 与服务器同步

远程仓库：[EricPang3/CVGL_NEW1117](https://github.com/EricPang3/CVGL_NEW1117)。源码、配置、固定地点划分和文档进入 Git；原始数据、权重、模型缓存、压缩包和 `outputs/` 实验产物由 `.gitignore` 排除。`outputs/.gitkeep` 仅保留空目录。数据与权重通过 AutoDL 文件上传或其他文件传输方式单独传送。

本地提交并推送后，服务器在项目目录执行 `git pull --ff-only` 更新代码；本地输出和数据不会随 Git 同步。详见 [使用教程的同步步骤](使用教程.md#6-github-本地与服务器同步)。

AI 后续维护先阅读 [AI_README.md](AI_README.md)，其中记录项目核心思想、模块契约、AutoDL 环境和验证边界。

环境与目录依据：[PyTorch 2.1.2 对应版本](https://dev-discuss.pytorch.org/t/pytorch-2-1-2-general-availability/1730)、[AutoDL 数据盘说明](https://www.autodl.com/docs/public_data/)。
