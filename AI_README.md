# CVGL_NEW：供 AI 维护者阅读

> 核对日期：2026-11-17。本文用于快速恢复项目上下文；行为以当前源码为准。

## 1. 核心目标

将“训练得到一个分数”的论文代码变为能追溯实验、查看逐图错误的简单工具。围绕三个独立入口组织：训练保存权重和记录；评估加载已有权重、导出排名和指标；报告读取结果、展示失败案例。保持少量直接的 Python 函数和数据类，不引入不必要的框架或多套 runner。

基线是 S4G 的 **共享 ConvNeXt-B + GAP + 双向 InfoNCE**。本项目保留架构和主要增强、采样约束，不宣称完整复现论文全部挖掘策略或公布指标。代码在 CVGL_EP 基础上精简改写，原项目保持独立。

## 2. 基线契约

- 模型固定为 `convnext_base.fb_in22k_ft_in1k_384`；两视角调用同一个 timm 编码器，不是两个独立分支。`num_classes=0`、`global_pool="avg"`；无新增分类器或投影头。
- `forward(image)` 返回 `[B,D]` 特征，`forward(sat, drone)` 返回两个特征张量。权重键名保留 `model.*` 与 `logit_scale`，兼容 CVGL_EP 的此类 state_dict。
- InfoNCE 内部 L2 归一化，相似度矩阵为 `[B,B]`，可学习尺度初值为 `1/0.07`。正样本为对角线，两个方向交叉熵取平均，默认标签平滑 0.1。
- batch 内地点必须唯一，避免同地点的其他图片变成假负样本。`UniqueBatchSampler` 保留完整 batch 和 512 次连续冲突停止规则，记录丢弃样本数。
- 数据增强保留旧项目的同步水平翻转、图像压缩、颜色扰动、模糊/锐化、遮挡；卫星路增加随机 90 度旋转。
- 训练始终是 satellite/drone 配对。`direction` 只记录默认评估方向，不能据此改变训练配对或强制分别训练。

## 3. 文件职责

| 文件 | 职责 |
|---|---|
| `configs/u1652.yaml`、`sues200.yaml` | 实验默认参数 |
| `configs/sues200_split.yaml` | CVGL_EP 原始固定 120 个训练地点；应随 Git 保存 |
| `train.py` | 参数、模型与 DataLoader 初始化、训练记录、checkpoint 保存 |
| `evaluate.py` | 独立加载权重、恢复和覆盖配置、保存评估产物 |
| `report.py` | 复制缩略图，内嵌数据，生成离线 HTML 和 ZIP |
| `check_env.py` | AutoDL 框架版本、GPU 与 torchvision CUDA 算子检查 |
| `sample4geo/model.py`、`loss.py` | 基线模型与损失 |
| `sample4geo/dataset/university.py` | 两数据集共用读取、配对、地点唯一采样 |
| `sample4geo/dataset/transforms.py` | 原增强及评估预处理 |
| `sample4geo/trainer.py` | 单轮优化、按实际 batch 进度的 warmup/cosine |
| `sample4geo/evaluate/` | 提特征、分块打分、排名、AP 和 Recall |
| `sample4geo/utils.py` | 配置、路径、随机种子和权重加载 |
| `tests/test_core.py` | 关键数据与指标语义测试 |

U1652 的文件名和类名被 SUES 共用是有意保留的复用关系，不需要仅为统一命名而拆分或重命名。

## 4. 目标运行环境与路径

正式训练：AutoDL Ubuntu 22.04，base 自带 Python 3.10、PyTorch 2.1.2、CUDA 11.8，单张 RTX 4090；对应 torchvision 0.16.2。默认 384 像素、24 对图/batch、FP16 AMP，评估 batch 64；不含 DDP/DataParallel。

`requirements.txt` 安装额外依赖，并引用 constraints 固定 torch/torchvision，复用服务器已有框架。无需为了本项目创建新 conda 环境或重装 torch。若需改变版本，先验证这套旧 timm/Albumentations API 的兼容性。

本地数据默认 `~/datasets/{U1652,SUES200}`；AutoDL 推荐 `/root/autodl-tmp/datasets/{U1652,SUES200}`。路径选择顺序：

1. 显式 `--data-root`（某一个数据集的根目录）。
2. `CVGL_DATA_ROOT/<数据集名>`（当前机器的统一数据父目录）。
3. YAML 或 checkpoint 中保存的 `data_root`。

这个优先级同时作用于训练和独立评估，跨机器移动权重时可以覆盖原绝对路径。环境变量不写进 Git，不改写共享配置。

U1652 读取已有 `train/`、`test/` 目录。SUES 读取原始 `satellite-view/<id>/` 和 `drone_view_512/<id>/<高度>/`，不复制数据；120 个训练地点，其余 80 个作查询，完整图库为 200 地点。读图返回原始地点和路径，不能用丢失身份的标签替代原始清单。

缓存推荐 `HF_HOME=/root/autodl-tmp/huggingface`、`HF_HUB_CACHE=$HF_HOME/hub`，由用户在服务器设置。代码和 outputs 可直接放在数据盘的项目目录中。

## 5. 训练、权重与评估产物

- `outputs/<name>/config.yaml`：训练实际配置、训练地点和 Python/torch/torchvision/CUDA/cuDNN/GPU 信息。
- `train.csv`：每轮 loss、lr、耗时、实际配对数、步数、丢弃数和尺度。
- `model.pth`：字典 `{model: state_dict, config: 实际配置, epoch: 已完成轮数}`。每轮通过临时文件原子替换，仅保留最后完成的一轮。
- 不根据测试集选最佳模型；训练不自动评估；不保存优化器，因此 `--init-checkpoint` 不是精确断点续训。
- 旧原始 state_dict 和 `module.` 前缀会被兼容；模型加载使用 strict 检查。独立评估 `pretrained=False`，无需下载初始化权重，也不读取训练目录。
- `evaluation.yaml`：本次实际评估配置、权重 SHA256；不覆盖已有训练 config。输出到新目录时补充该次的 config。
- `metrics.json`：指标和数量、口径。`queries.json` 包含 `{evaluation, gallery, queries}`；图库清单保留 included 标记，每张查询保存 AP、首个正确排名、正样本数量、Top-K、最佳正确候选。
- 不落盘特征缓存和完整 Q×G 分数矩阵。特征提取后暂存 CPU，图库特征用于当前设备打分；评分按查询块计算。
- 已有评估文件默认拒绝覆盖，用新 `--output` 或显式 `--overwrite`。

## 6. 指标口径：不要静默改变

JSON 指标为 0–1，终端/报告显示百分比。正确匹配为地点 ID 相同，支持多个正样本。FP32 归一化后计算余弦相似度，同分按已排序路径稳定排序。

默认 `gallery_scope=all`，保留额外干扰地点；`query_ids` 仅兼容 CVGL_EP 的查询地点过滤。标签 `-1` 不参与排名。AP 保留旧 `compute_mAP` 的梯形积分，不等于 sklearn 阶梯式 AP。R@1% 使用有效图库的 `max(1, ceil(G*0.01))`；无正样本查询保留 null 记录，从指标分母排除并计数，全部无匹配则报错。

与 CVGL_EP 的差别：默认完整图库、R@1% 修正、无正样本分母、FP32 归一化和同分处理。训练还改变了默认轮数（5 对旧 1）、样本顺序、调度推进和保存策略。因此基线架构相同不代表训练权重或正式分数逐位相同；比较时必须控制权重、划分、图库范围和协议。指标协议版本当前为 `cvgl_new_v1`。

## 7. 报告契约

报告生成依赖 `metrics.json`、`queries.json` 和源图片；生成后 `report/index.html`、`report/images/` 及 `report.zip` 可离线使用。无 CDN、fetch、外部字体或服务端依赖。

默认最多 200 个案例，无正样本和排名靠后优先；`--limit 0` 导出全部。筛选和搜索只作用于已收录案例，不影响 JSON 或指标。失败定义为有效查询的 Top-1 错误；无正样本另列。HTML 内嵌 JSON 转义 `<>&`，用户文本用 textContent，保持离线和安全输出方式。

## 8. GitHub 与维护约定

仓库：`https://github.com/EricPang3/CVGL_NEW1117`。Git 同步源码、默认配置、划分文件、测试和文档。数据集、模型权重、缓存、压缩包、outputs 实验产物由 `.gitignore` 排除。不要强制添加这些文件，不将服务器绝对路径改写进公共 YAML，也不要清理用户已有数据或实验目录。

做修改时保持范围具体。模型、损失、采样、指标协议的变化要明确说明并验证；配置和 CLI 的变化同步更新 README 与使用教程。避免为了名称一致或未来可能扩展而增加抽象层。文档中的历史测试不等于本轮已执行测试。

## 9. 验证与已知边界

基本检查：`python -m unittest discover -s tests -v`。新增 AutoDL 数据根目录优先级测试后共 7 项，验证 CLI > 环境 > 默认路径；其余覆盖排序/AP 边界、采样、反传、SUES 划分、离线报告。

历史验证：真实 ConvNeXt-B 在 32×32 合成数据完成一次 CPU 更新、checkpoint 保存、独立双向评估、旧权重加载和报告打包；同权重旧/新前向与损失数值对齐。真实数据只扫描过路径和划分。报告脚本做过 DOM 模拟，未做真实浏览器渲染。

当前开发环境是 CPU，不是用户的 AutoDL 服务器；不能声称已验证 4090 显存、速度或正式训练分数。服务器由用户运行 `python check_env.py` 检查真实 GPU 环境。不要在未执行实验时填充或推测实验分数。
