# 研究状态与续接记录

更新日期：2026-11-22（北京时间）。此文件可维护；手写核心思想文件不可修改。

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

## 新一轮实现与计划（GPU新指标尚未产生）

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

## 当前阻塞及下一步

- 旧端口22352已停止。用户提供新SSH：connect.bjb1.seetacloud.com:43322；已连接并部署研究分支，数据完整，GPU预检完成后4090空闲，数据盘约19GiB可用。密码不得写入本文件；本地复用socket为/tmp/cvgl-ssh-43322。
- 长队列预算仍待用户选择（24/48小时或仅准备）；此前已异步询问。先部署和GPU预检，不擅自将无回复当作预算批准。
- 本地新分支experiments/20261122-research-screen；6bf40ac单独保存上一轮最终表格。新候选尚无正式GPU训练/测试指标。
- c24a718已推送GitHub同名研究分支；AutoDL原Python3.10/torch2.1.2+cu118环境48项完整回归通过。部署记录和服务器测试日志在outputs/research20261122_preflight；后续接收策略指纹修正有单独Git提交，不改变模型/训练公式。
- 当前尚未启动research20261122长队列。预算确定后按README_RESEARCH.md启动服务器驱动与本地接收；不需要再次接模块。全流程使用同提交部署，具体SHA查Git和outputs/research20261122_preflight/deployment.json。旧中午截止不可复用。
- 本文件中的“计划”须随实际完成更新，维护实际代码提交、验证日志、活动队列名与未解决问题，保证上下文丢失后能继续。
