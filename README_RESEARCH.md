# 自动模块实验：核心思想与运行说明

更新：2026-11-22。续接顺序：`核心思想（AI必读）.md`（不可修改）→`AGENTS.md`→`RESEARCH_STATE.md`→本文件。真实运行状态以对应`outputs/<run>/state.json`为准。

## 要实现什么

以共享ConvNeXt-B＋GAP＋双向InfoNCE的S4G为固定起点，寻找能在U1652、SUES200无人机—卫星检索上稳定提高精度的设计。先筛查是否涨点，有提升才深入分析修好/退步的案例。最终目标是同协议超过近期强方法；接入模块、预检通过、单次最高分均不等于达到目标。

代码、配方、来源、汇总表进Git；数据、权重、完整queries和HTML留在outputs，不进Git。手写核心文件保持原样，可维护文档记录最新授权、事实、假设与下一步。

## 本轮的16个候选

| 类别 | 候选 | 借鉴思想及对应控制 |
|---|---|---|
| 部件 | prototype_pool / prototype_graph | SkyPart启发的共享原型；比较普通池化与部件图 |
| 部分对应 | prototype_dustbin | GenGeo/SALAD启发的额外丢弃原型；本实现是softmax近似，无最优传输 |
| 语义尺度 | scale_gate / scale_uniform | DINO-GFSA门控启发；同一特征图的3种膨胀卷积，学习选择对比均匀融合 |
| 频域 | frequency_gate / spatial_smooth | 自写频段门控与空间平滑对照；经过非线性再GAP，防止高频在线性GAP中恒为零 |
| 组合 | wt_proto_graph / dw_proto_graph | WTConv＋部件图，对照普通深度卷积＋同一部件图 |
| 训练目标 | consensus / visible_consensus | GeoCoM启发的训练期联合锚点；可见性仅为语义相似/部件占比代理，无深度几何教师 |
| 训练目标 | part_align / view_relation / vicreg | 部件跨视角InfoNCE、地点关系一致性、描述子统计约束 |
| 组合 | wt_consensus / ppm_consensus | 已接入特征分支分别组合训练期共识 |

来源与差异详见`configs/research_sources.yaml`，近期强方法原报指标见`results/literature_20261121.csv`。本轮实现保留ConvNeXt和GAP旁路，没有换成DINO骨干，也没有复制完整SkyPart/GeoCoM/MGS²。频域、关系和统计目标是自写假设/通用约束，不能冒充新论文的完整复现。

默认`module=none, auxiliary=none`仍走旧编码器和原InfoNCE路径，权重名兼容；显式辅助分支只在训练读取成对特征。评估时每张图独立编码，绝不传入正确配对。

## 如何决定继续跑谁

`configs/research_campaign.yaml`固定候选顺序、超参和晋级规则，不根据官方测试成绩自动调参。

1. 在原训练地点中固定留20%开发集，split seed20261122，与模型训练seed分开；所有候选共用。U1652为561训练/140开发地点；SUES为96训练/24开发地点，四高度使用同一组ID。
2. 开发筛查：baseline＋16候选，各训练U1652和SUES150m，2epoch、物理batch24。每个权重评估双向，共34次训练。
3. 只比较开发集四个方向/任务的平均AP变化。至少+0.2pp、每个数据集双向平均R@1退步不超过1pp、任务完整，才进入前3候选复验。
4. 复验用seed2，同样2epoch和同预算baseline。合并两seed仍达到门槛，且每个seed的平均AP变化均为正，至多前2晋级。没有通过者就如实结束，不通过刷官方测试挑赢家。
5. 固定最终矩阵：回到全部原训练地点，每个晋级候选与baseline都训练3epoch；U1652＋SUES四高度，seed1/2/3，双向。SUES额外分别记录`query_ids`与`all`图库。最多45次最终训练，预算不足则保留未完成状态，不填补估计值。

筛查与最终矩阵的epoch数不同，最后必须比较同为3epoch、同seed的baseline；不能拿新3epoch结果减旧1epoch基线冒充模块增益。最终测试的正负结果都保留，不按测试成绩追补超参。开发集控制实验只支持开发阶段的判断；若某机制对照未晋级完整矩阵，不能据此宣称其在完整协议中已被证明。

SUES主协议仍为CAMP/DAC的固定120/80地点划分。SkyPart的顺序1–120/121–200划分与本项目不同，图库也可能不同；不能直接换其测试ID评估本项目已有权重后横比。平均跨任务开发分仅用于筛查，不是官方指标。

## 运行与续跑

环境沿用AutoDL：Python3.10、torch2.1.2+cu118、单RTX4090；不升级torch，不新增大型框架依赖。数据父目录为`/root/autodl-tmp/CVGL_NEW/data`，含`U1652`和`SUES200`。

先查看队列，命令不训练、不创建实验状态：

```bash
python scripts/research_campaign.py --plan
```

预算确定后在服务器启动；以下24小时是示例，48小时用`--hours 48`：

```bash
cd /root/autodl-tmp/CVGL_NEW
nohup /root/miniconda3/bin/python scripts/research_campaign.py \
  --hours 24 --shutdown > outputs/research20261122.launch.log 2>&1 < /dev/null &
```

先自动做GPU预检，再运行。队列有全局研究GPU锁；重复启动会拒绝。预算按从首次启动起的壁钟时间计算，包括评估/归档，截止后不开始新任务，当前任务完成后结束，因此可能超过名义预算一个任务的时间。不会因重启而重置截止时间。

同配方中断后使用相同命令续跑。已完成任务不重训；未完整完成的训练移入`*_interrupted_*`保留，重新从相同初始化训练，不假称恢复了优化器。修复代码后用`--repair-note '具体原因' --retry-failed`，旧源码、失败日志与新源码hash都保留；这不会延长原预算。改配方或预算需新运行名，不能覆盖原实验。

单项失败记录后继续其他候选；baseline失败则停止对应比较。脚本不具备自动写代码修bug的能力，AI应检查错误、修复并登记后续跑。`partial_failure`与`budget_reached`不表示完整矩阵成功。

独立使用新入口示例：

```bash
python train.py --dataset U1652 --name proto_dev --module prototype_graph \
  --auxiliary consensus --auxiliary-weight 0.5 --development-fraction 0.2 --epochs 2
python evaluate.py --checkpoint outputs/proto_dev/model.pth \
  --split development --direction D2S --output outputs/proto_dev/D2S
```

开发集模型不等于最终模型。正式训练用`--development-fraction 0`；正式评估默认`--split test`，不需要训练图片。

## 结果、权重和离线可视化

- `outputs/research20261122/`：state.json、protocol.yaml、commands.jsonl、metrics.csv、逐任务日志、source.zip、archive_ready与逐模型tar.gz。
- `outputs/research20261122_<阶段>_s<seed>_<候选>_<任务>/`：config.yaml、train.csv、source.zip、权重，以及各方向metrics/queries。SUES完整图库结果位于`all/D2S`与`all/S2D`。
- 所有成功任务先校验归档，落选权重才按策略清理；删除记录含SHA、字节数、原因和归档SHA。baseline权重受保护，最终seed1模型保留；之后seed的候选保留完整结果再释放权重。开发阶段滚动保留前4个候选的权重，晋级名单固定后释放开发候选权重；最终模型重新从ImageNet在完整训练集训练。
- 小于3GiB可用空间时记录错误并停止，不等本地接收腾空间而长时间空转。不会删除不属于本队列的权重。
- 服务器校验归档后移除重复的解压报告、report.zip和非baseline的原始queries副本，完整文件均在对应tar.gz中，见PACKED_RESULTS.json。baseline原始queries保留供同任务比较；本地接收会解出所有文件，案例功能不受影响。后续权重清理重写归档时会合并保留原包中的查询/报告。

本地建立SSH复用连接（密码在终端输入，不写入命令/配置文件）：

```bash
ssh -MNf -S /tmp/cvgl-ssh-43322 -o ControlMaster=yes -o ControlPersist=12h \
  -o UserKnownHostsFile=/tmp/cvgl_known_hosts -o StrictHostKeyChecking=accept-new \
  -p 43322 root@connect.bjb1.seetacloud.com
python scripts/collect_experiments.py --host root@connect.bjb1.seetacloud.com \
  --port 43322 --socket /tmp/cvgl-ssh-43322 --run research20261122
```

本地接收会校验SHA再解包，下载最终seed1权重，生成`outputs/research20261122/index.html`。点击有链接的行看离线案例，可筛选修好/退步；原始数据不在本地也能看。开发阶段未达到增益门槛的项不生成深入页面，完整逐图数据仍保留。SUES两个图库各有独立页面。

可入Git的结果在`results/research20261122/metrics.csv`、`selection.csv`、`paired_seeds.csv`；按阶段、seed、数据集、高度、方向、图库分列。配对均值/标准差只在同一任务内计算，n不足时如实显示，不把单次涨点写成稳定结论。

`--shutdown`会验证服务器成功任务的全部归档，最多等本地确认10分钟后关机；本地不在线也会按期结束，结果留在AutoDL数据盘。若关机前没收齐，下次开机更新SSH端口后重启接收。请勿释放实例或删除数据盘后再找未下载产物。关机和本地归档是两个单独记录的事实。

## 当前验证范围

2026-11-22已在真实数据核对五任务地点互斥与清单指纹。RTX4090上17个全模型（baseline＋16候选）全部通过B24/384、含LR0到0.001的4步优化检查；14个模块/辅助目标通过幅值1/32的特征图有限值/梯度检查。最大训练显存约19630MiB。48项回归已通过，包含阶段转换/双图库/归档/清理/续跑的队列贯通测试；最终验证状态见RESEARCH_STATE.md。

用户再次要求继续后，队列已于北京时间2026-11-22 12:20:47启动，采用推荐24小时上限；2026-11-23 12:20:47后停止新增任务，当前任务归档后关机。本地接收同时运行。启动时尚无本轮精度指标；继续时先读state.json和维护记录，不把预检当作精度结果。
