# CVGL_NEW：AI 维护契约

> 2026-11-22最新任务：继续自动模块筛查并对标近期强方法，用户偏好先涨点、后分析。续接先读AGENTS.md与RESEARCH_STATE.md。新服务器43322已连接，数据/4090可用；长队列24/48小时预算待用户选择。原baseline契约只约束显式none路径，新候选允许训练期辅助分支与已记录的头部适配；新队列configs/research_campaign.yaml，思想来源configs/research_sources.yaml。

> 2026-11-20：用户授权33模块完整矩阵。先读不可修改的《核心思想（AI必读）.md》、模块目录和EXPERIMENT_LOG.md。以下baseline契约适用于module=none；显式模块实验允许目录中的共享适配头，不能删除原GAP旁路。队列配置configs/module_suite.yaml，脚本scripts/module_suite.py，本地接收scripts/collect_experiments.py。真实完成状态看outputs/modules20261120/state.json，不要把已排队写成已完成。

## 目标与用户约定

目标是从“训练得到分数”变成可追溯、可看失败、可对照改进的实验工具。三个业务入口：train.py、evaluate.py、report.py。scripts/run_experiments.py只是顺序调用和打包，禁止再实现训练/指标逻辑。

用户决定两个数据集batch默认24。目标AutoDL Ubuntu22.04、单4090、base Python3.10 / torch2.1.2+cu118 / torchvision0.16.2。不要无故升级torch。仓库 https://github.com/EricPang3/CVGL_NEW1117；数据、权重、缓存、outputs不入Git。

## 算法契约

- timm `convnext_base.fb_in22k_ft_in1k_384`，两路共享、num_classes=0、GAP，无新分类头/投影头/分块/域适应。保留model.*和logit_scale权重名，兼容同结构旧state_dict。
- 双向InfoNCE，归一化特征，对角线正样本，两向CE平均、平滑.1，learnable scale初值1/.07。
- 默认24对、1epoch、seed1、384、AdamW .001/.01、warmup .1、FP16初始scale1024、按值梯度裁剪100。batch24不同于原S4G U1652 batch128；普通梯度累积不增加InfoNCE对比集合。
- U1652 augmentation=s4g；SUES augmentation=camp：卫星contrast/saturation/hue=.3，无人机contrast=.7、saturation/hue=.3，brightness均.15；其他变换保留S4G。
- 同步翻转用NumPy；sampler持续用全局Python RNG，不逐epoch重设seed，不再首轮seed+1。完整batch、512次连续冲突停止规则不可改。显式独立seed只用于对照/测试，训练不传。
- scheduler与原HF cosine公式相同，首步LR0，optimizer后调用scheduler。计划步数使用采样前ceil(N/B)，不是采样后loader长度；AMP跳过仍按原入口推进并记录跳过数。
- 文件顺序稳定排序，但不承诺原os.walk次序或GPU逐位确定性。
- 训练配对不受检索方向影响；同权重评双向，四高度独立初始化。保存最后一轮，不用测试集选checkpoint。init-checkpoint只载参数，不是完整续训。

## 数据与评估

- 路径优先级CLI单数据集根 > CVGL_DATA_ROOT父目录 > YAML/checkpoint。当前服务器 `/root/autodl-tmp/CVGL_NEW/data/{U1652,SUES200}`；本地默认~/datasets。
- SUES直接读原图；固定120训练ID与本地CAMP/DAC相同，余80查询，原始图库200ID。不得随机重划或静默混高度。
- 主图库query_ids，与参考代码标-1后排除范围一致；all是额外干扰实验，不可混用比较。
- key是相对数据集根路径，path是实际图片位置。清单hash覆盖key+ID，不验证图片字节。
- 独立评估不能训练、下载预训练或访问训练图片。training_config.yaml保留权重原始元数据，evaluation.yaml记录本次测试设置；旧权重不可伪造训练历史。
- FP32归一化/相似度，编码器可AMP；同分按稳定路径排序。AP用S4G梯形积分。R@1%仍是ceil(有效G/100)，未照搬原索引偏移。无正样本排除并记录。JSON0–1，页面/终端/汇总CSV百分比。

## 本地分析契约

- report.py只依赖标准库和Pillow；浏览HTML无需任何Python/GPU/原始数据，无CDN/fetch/外部服务。
- 默认打包全部缩略图，ZIP含index.html、images、完整queries、指标、配置、训练CSV、源码快照；不含权重/原始分辨率图片。限量报告不能离线展示未打包的图片。
- 本地重建优先复用下载缩略图。比较前核对任务、预处理、查询、有效图库和协议；只允许训练配方不同。输出修好/退步和comparison.json，不能把图库变化当改进。
- 人工标注浏览器保存并导出JSON；file://的localStorage不保证可用，导出是正式保存方式。HTML用textContent插入内容，嵌入JSON必须转义<、>、&。
- 案例用于提出假设，不能自动证明原因。新研究队列在训练地点内显式留20%开发集；正式矩阵回到完整原训练地点。开发划分ID/hash进入配置，不能用旧的全训练权重当无泄漏开发集模型。

## 运行边界

2026-11-21用户明确允许删除不用权重。完整首轮370项已完成，PAM已统一修复重跑；results/保存可入Git的CSV表，outputs/保留原始指标、queries、离线报告、配置、源码和归档记录。configs/checkpoint_retention.json只控制存储，不筛选训练任务。保留各seed baseline、任务最优候选和关键对照；其余权重可按策略删除，所有轻量reranker头保留。先检查results/checkpoint_inventory.csv和实际文件是否存在，不要假设每个已评估模型都有model.pth。旧20261117及smoke权重已按用户授权删除，旧指标单独存legacy表。

scripts/experiment_records.py导出百分比表与配对seed表；--apply仅在记录完整时清理已完成的本地权重，清理前停止接收程序以免并发写manifest。删除写weight_cleanup.csv，future不下载的权重写manifest的checkpoint_storage=discarded_by_policy及策略SHA。离线案例不需要权重；重新提特征需要保留权重或按原配置重训。

当前夜间流程是run_overnight.py与collect_overnight.py：原矩阵完成后补seed复验，12点后停止新增补充任务，当前任务完成且所有记录校验归档后关机。module_suite.py会记录失败并继续其他任务；只有已经验证并登记的repair_request才整组重跑，不具备自动改代码能力。下面的失败停止规则特指旧批量入口run_experiments.py。

批量入口默认5训练/10评估/10报告；仅全部成功并打包后按显式--shutdown关机。失败落run.json和日志并停止，保留产物，不自动续跑或自动修复。AutoDL /usr/bin/shutdown是无shebang脚本，此环境用/bin/bash调用。

源码快照和hash、实际配置、库版本、地点/清单/权重指纹用于追溯。旧20261117_full_seed1属于旧配方，不覆盖、不重命名成新结果。代码修改不等于GPU新指标验证完成。

来源核对：本地Sample4Geo commit498c651、CAMP b04a9c8、DAC 5612a79。沿入口实际import追踪实现；CAMP/DAC的SUES入口复用dataset.university。U1652训练行为参考S4G，SUES数据/增强/batch24参考CAMP/DAC，不复制其方法模块。

修改后运行 `python -m unittest discover -s tests -v`。训练语义、协议、离线迁移值得回归测试，文案无需镜像实现测试。报告CPU/GPU验证的真实范围。
