# 研究状态与续接记录

更新日期：2026-11-23（北京时间）。此文件可维护；手写核心思想文件不可修改。

## 当前用户授权

继续独立寻找近期可能有效的模块，接入并自动实验；通过README等文件保存核心内容。偏好是“先看是否涨点，有提升再分析”。目标是同协议超过近期强方法，而不止超越旧baseline。

## 已完成与事实

- 工作仓库 `/home/ericp/projects/CVGL_NEW`，GitHub EricPang3/CVGL_NEW1117；不要修改同级项目。
- 首轮33模块+4对照，370项评估全部完成；PAM因FP16点积溢出改为FP32，五任务统一重跑完成。另有baseline十项。
- seed2受旧中午预算限制完成78项模块评估，未完成第二轮完整矩阵。两轮均已校验归档，2026-11-21约12:01请求关机。原deadline不再适用于新一轮。
- 主表在results/，逐图/报告/源码/配置在outputs/。已授权删除无用权重，清理本地143份约46.93GiB、远程7份旧权重约2.28GiB；不要假设每个模型权重仍存在。
- 首轮U1652 D2S最好ACB92.606%，原基线92.408%；S2D R1最高Patch-NetVLAD适配版95.720%，PPM更均衡（R1 95.578%，AP91.717%）。这些是探索性结果。
- U1652 D2S两seed相对各自基线：WTConv +0.098/+0.143pp；burst +0.148/+0.153pp；ECA +0.135/-0.365pp。尚未证明普遍稳定改进。
- 已核对GeoCoM、MGS²-Net、SkyPart等近期论文，原报指标/协议/代码状态保存在results/literature_20261121.csv。GeoCoM当前只有README；MGS²代码链接为空。自写适配不能冒充作者完整复现。
- SkyPart文档的SUES训练1–120/测试121–200以及full200图库，与本项目CAMP划分/query_ids不同，不能直接混比分数或只换测试集评旧权重。

## 新一轮实现与运行

1. 已实现训练地点20%开发集，固定split seed20261122与ID/hash；默认官方独立评估仍不访问训练图片，显式--split development才使用开发集。
2. 已接入9种特征头/对照：prototype_pool/graph/dustbin、scale_gate/uniform、frequency_gate/spatial_smooth、wt_proto_graph/dw_proto_graph。均保留GAP旁路。
3. 已接入5种训练期目标：consensus、visible_consensus、part_align、view_relation、vicreg。推理不调用成对分支；共识与可见性均为轻量思想适配，不等于原论文完整实现。
4. configs/research_campaign.yaml固定16候选（含WT/PPM+共识组合）与baseline；开发集2epoch筛查→seed2复验前3→至多2个候选的5任务×3seed、3epoch完整矩阵。同预算baseline配对；官方测试不驱动晋级。
5. scripts/research_campaign.py已实现预算/锁/续跑/命令留痕/完整归档后服务器自主清理，不依赖本地在线才继续。正负结果均保留，深入报告仅开发集正增益与固定最终矩阵。
6. 完整48项回归通过（原40＋新增8），覆盖真实特征图、大幅值有限梯度、辅助分支独立推理、开发集泄漏拒绝、只凭开发集晋级、预算、归档清理及完整队列贯通。贯通测试模拟昂贵训练命令，真实执行比较/报告/归档/清理/续跑，不能当作精度结果。
7. 服务器RTX4090预检已通过：14项特征图/辅助目标幅值1/32有限梯度检查，17个全模型（baseline＋16候选）物理B24/384四步优化检查均通过，最大19629.55MiB。算法预检提交33878df，JSON已在本地/服务器outputs/research20261122_preflight/保存。
8. 真实数据五任务开发划分已核对：U1652训练561地点/30294对、开发140地点/7560无人机查询；SUES每高度训练96地点/4800对、开发24地点/1200查询。ID/hash保存同目录data_protocol.json。
9. 本地接收沿用scripts/collect_experiments.py，自动识别cvgl_research_v1；分阶段导results/research20261122表格，完整图库另页，最终seed1权重下载。后续seed候选只保留完整结果。开发晋级结束后释放不再被引用的开发候选权重，baseline保护。
10. 最终空间检查发现报告解压目录会再复制完整queries（旧U1652 D2S约128MiB）。新驱动校验tar.gz后去掉重复报告/zip与非baseline原始queries，完整内容留包、本地正常解包；baseline queries保留供比较。重写权重删除记录时合并旧包内容，不能丢掉已收纳的报告。

## 当前运行与续接

- 旧端口22352已停止。当前SSH：connect.bjb1.seetacloud.com:43322；数据完整，启动前约19GiB可用。密码不得写入本文件；本地复用socket为/tmp/cvgl-ssh-43322。
- 2026-11-22用户再次明确要求“继续跑实验”，已启动此前推荐的24小时有限队列，无需继续等待预算确认。运行research20261122，北京时间11月22日12:20:47启动，11月23日12:20:47后不开始新任务；当前任务完成并验证归档后关机，若矩阵提前完成则提前结束。不是保证恰在截止时刻关机。
- 本地新分支experiments/20261122-research-screen；6bf40ac单独保存上一轮最终表格。本轮16候选初筛、seed2开发复验以及3seed官方矩阵均已完成；不把开发分数与论文官方分数混比。
- c24a718已推送GitHub同名研究分支；AutoDL原Python3.10/torch2.1.2+cu118环境48项完整回归通过。部署记录和服务器测试日志在outputs/research20261122_preflight；后续接收策略指纹修正有单独Git提交，不改变模型/训练公式。
- 本次算法与编排提交dd015ff已同时在本地、GitHub、服务器；只更新维护文档不改变state中的code_sha256。初始服务器驱动PID3017、参数--hours 24 --shutdown；初始本地接收PID162552。这些PID仅为历史记录，本轮已complete，不应据旧数字重新启动。启动与接收日志仍保留。
- 最新状态以服务器outputs/research20261122/state.json与本地接收副本为准；启动时尚无本轮精度指标。启动预检通过后按固定顺序执行34次开发训练及按规则晋级的复验/完整矩阵。发现failed时先看逐任务日志并保留原记录，再登记repair-note续跑；不要重开过期夜间计划，不要重复启动持锁队列。
- 本文件中的“计划”须随实际完成更新，维护实际代码提交、验证日志、活动队列名与未解决问题，保证上下文丢失后能继续。

## 2026-11-22 13:46 检查快照

- 队列running，完成11/34组开发筛查任务（含2组baseline），已有22项方向评估；4个候选已完成双数据集，第5个visible_consensus正在训练SUES150m。驱动与本地接收均存活，11包均已本地归档，无接收错误。磁盘约15GiB可用，无失败任务，已完成训练AMP跳步均0。
- 本轮开发集baseline：U1652 D2S R1/AP97.2222/97.6941，S2D95.7143/96.8663；SUES150m D2S97.9167/98.3993，S2D100/97.2706。均2epoch、seed1，非官方测试。
- visible_consensus在U1652：D2S98.7566/98.8661（R1+1.5344pp、AP+1.1719pp），S2D98.5714/98.1926（R1+2.8571pp、AP+1.3263pp）。SUES尚在跑；不能据此宣布跨数据集稳定有效或SOTA。
- 完成四个方向的候选开发平均AP变化：consensus+0.9508pp，prototype_pool+0.9788pp，prototype_graph+0.5166pp，prototype_dustbin+0.1862pp。graph在U D2S R1-0.9259pp，dustbin在SUES S2D R1-4.1667pp；保留负结果，不只报局部最好项。候选顺序和晋级规则不变，尚未开始seed2复验。
- 不可变CSV快照results/research20261122/status_20261122_1346.csv；持续更新主表为metrics.csv。完整原始/案例在outputs/research20261122/artifacts与index.html。

## 2026-11-22 19:27 正式测试检查与配方问题

- 服务器running：34/34组开发筛查、8/8组seed2开发复验完成；正式矩阵13/45组完成，总计55组完成、130项评估。当前final_s1_visible_consensus_sues200_300/train。无失败，数据盘约10.73GiB可用，截止仍为11月23日12:20:47后停止新增任务。
- 复验通过visible_consensus与wt_consensus；prototype_pool开发seed1平均AP+0.9788pp，seed2-0.5711pp，不满足各seed均为正的规则，未晋级。visible两开发seed平均AP+0.7840pp、WT共识+0.7433pp；这些是开发分数，不是官方测试。
- 官方U1652 seed1、3epoch、query_ids：baseline D2S R1/AP90.2787/91.7819，S2D93.5806/89.2782；visible90.8440/92.2795、94.1512/90.3122；WT共识91.5440/92.8462、93.5806/90.4195。当前相对本轮baseline有局部增益，但都未超过旧1epoch baseline的U1652 R1（92.4079/94.5792）。不能宣布已超过原baseline或强方法。
- SUES150m WT共识官方D2S R1从94.00到95.95（+1.95pp），S2D93.75到96.25（+2.50pp）；SUES200m其D2S/S2D R1反而-0.575/-1.25pp。SUES250m visible在query_ids D2S+0.525pp，在all图库D2S-0.55pp；不同图库分开报告，保留负结果。
- 已只读核对旧1epoch与本轮3epoch baseline配置：训练701地点完全一致，训练两视角清单指纹、查询/图库清单指纹、评估protocol均相同，关键训练配方差异是epochs1→3及对应cosine总步数。新增none/auxiliary/development_fraction为默认元数据，无有效分支。训练预算/调度可能导致退化，尚未证明过拟合或实现bug；不要据官方测试结果临时改正在跑的预注册矩阵。下一轮若验证轮数/调度须单列对照，不覆盖本轮结果。
- 本地接收存活、无记录错误，因下载最终seed1大权重，状态和案例归档可落后服务器数分钟。已观察model.pth.partial持续写入，不应因此误判训练停止。共享SSH复用连接查询曾延迟；已另建短期只读监控socket /tmp/cvgl-status-43322，过期需重新登录，不要随意杀主接收连接。
- 完整最新快照results/research20261122/status_20261122_1927.csv，服务器状态副本outputs/research20261122/observations/state_20261122_1927.json。当前仍只有seed1官方结果，后续seed2/3官方测试未完成。

## 2026-11-23 完成与全面复盘（优先读取）

- research20261122于北京时间00:04:05完成，壁钟11.72小时：87/87组、246项评估，failures为空，45/45组3seed正式矩阵全部完成。87个当前结果包均已本地校验归档，接收日志complete。原SSH43322端口拒接，与预设归档后关机流程一致，未取得独立AutoDL电源状态回执；不要为查表而重新开机。
- 总计714条当前有效记录：旧1epoch基线20、33模块＋4对照首轮370、部分seed2模块78、第二轮开发68＋复验16＋正式162。另24条旧配方历史单列，不混比；首轮seed2并未完成全矩阵。
- 本次逐项核对714条R1/mAP与原始JSON，重算54组三seed均值/样本标准差，核对新246条配对差值与协议/清单；321个归档包SHA256及第二轮15份保留权重校验均通过，无缺失或不一致。第二轮服务器按记录删除58份不用权重约19.016GiB，完整结果均保留；本地有213份第二轮离线案例页。
- U1652正式3seed均值：3epoch baseline双向R1为89.6738/93.7233；visible为90.8845/94.1512（+1.2108/+0.4280pp），WT共识91.2482/93.9610（+1.5744/+0.2378pp）。两方法D2S三个seed均为正；S2D visible有一个seed为负。不能把均值为正说成每次都涨。
- 主协议10任务中visible平均R1为7涨3降，WT共识6涨3降1平；200m两方法均退步。全图库另报，不与query_ids混比。尚无统一方法稳定覆盖双向、多高度和不同图库。
- 旧1epoch baseline的U1652两seed均值92.3894/94.5078，高于当前3epoch方案。只比较共同seed1/2，3epoch baseline的D2S分别下降2.1292/3.2334pp，平均-2.6813pp。20组共同seed任务的训练关键配置（除epoch）、清单与评估规则核对一致；epochs与对应调度长度是重点怀疑，尚未确定因果。下一步应在原1epoch配方下重新做匹配消融，不把相对弱基线的增益当成达到目标。
- 现有正式矩阵缺少同预算普通共识/仅WT/普通卷积＋共识全套消融，不能证明组合收益来自WT本身。单seed筛查、开发集、测试集、不同epoch、不同图库均须分别报告。
- 完整成果表results/review20261123/summary.md；全部714条明细all_results.csv；主协议均值/标准差/每seed胜负primary_comparison.csv；全图库三seed表three_seed_metrics.csv；20组同seed配方审计same_seed_recipe_audit.csv；完整性audit.json。此轮用户请求是复盘，不启动新GPU实验。

## 2026-11-23 新授权与一轮固定消融（当前状态优先）

- 用户要求“你认为接下来应当怎么修改，然后继续进行实验”，已实现新分支experiments/20261123-e1-ablation；不再受上一条“只复盘不训练”的任务范围限制。
- 采用原1epoch/batch24/384配方：baseline、WT、精确可训练参数匹配DW、普通共识、可见共识、WT共识、DW共识。7组事先固定，负结果对照不删组，官方测试不调参；5任务×3seed，共105组正式训练。
- 额外在U1652/SUES150m开发集做1轮调度、3轮调度真实第1轮快照和第3轮诊断，2次额外训练/2个导入checkpoint。全轮上限135次实际训练、137组产物、442项评估，24小时任务边界停止并验证归档后关机。具体见README_ABLATION.md。
- 新增checkpoint_epoch/SHA与不可覆盖的原子快照；导入第1轮只拷贝真实前缀记录并记零额外训练。默认baseline路径、参数名、学习率公式保持兼容。
- 本地完整53项CPU回归通过，包含真实小模型3epoch轨迹和独立开发评估、负结果对照晋级固定矩阵、归档/清理/本地接收/续跑；4种配置内特征图/目标在幅值1/32通过CPU有限梯度检查。不是4090预检或新精度。
- 当前SSH connect.bjb1.seetacloud.com:43322仍Connection refused。已异步请用户开机并提供当前SSH，密码不得落盘。新队列尚未部署/启动，没有新PID/新分数，也不从本地准备时开始消耗24小时预算。
- 空间准备：仅计划回收已校验本地完整归档与权重SHA的远程重复文件；本地baseline和服务器唯一副本保留。清单在outputs/ablation20261123_e1_preparation/remote_duplicate_plan.json，目前未在服务器执行。恢复SSH后须核对实时GPU、数据、空间与同源码，做完整7模型B24 CUDA预检，再启动新队列和本地接收。
- 新结果目标目录outputs/ablation20261123_e1/、results/ablation20261123_e1/。额外导出diagnostics.csv、ablation_effects.csv、ablation_summary.csv，完整性/n/正负seed分列。旧714条审计结果和手写核心思想文件不覆盖。
