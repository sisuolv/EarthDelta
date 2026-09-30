# T5：条件性后续，尚不排期

状态TO_BE_RUN且未授权。只有T4实际结果、人工Gate1、A5及新的数据角色/预算签署后，才细化并执行。本文件保留研究约束，不是让代理接着开工。

保留的比较：F0、OBC、DABC、OCL-fresh、训练好的静态Fs*、sliding_refit与online_lora。与滑窗重拟合对齐样本曝光、优化步数和实际卡时；Fs*选择期不如F0时允许预定回退。所有臂共享合法反馈、各自持有adapter/optimizer/RNG/保护门/消费游标，发布不可回写。

如果Gate1支持输出路线而参数没有增量，优先研究输出订正与近期重拟合，不默认开发更复杂LoRA。若确需online_lora，最多一个主配置加6个单因素变体，不做全因子搜索。

LoRA复用已有build_fs_adapter/ExpertLoRA/controlled_rollout。共有backbone串行，无外部checkpoint；明确遗忘作用在参数还是ΔW上、optimizer状态是否保留、每记录消费次数、保护门仅看已兑现数据。旧设计的因子收缩不能未经推导就宣称等于ΔW的指数遗忘；在新T5协议中写清后再实现。

batch5（TAFAS/OnlineTSF）在本任务设计定案前完成，仍需A2b，核验HEAD/日期/license，不拷无license或NC/SA代码。PETSA的后缀回写评价不可迁入不可回写发布协议。

2020分析期已经用于Gate1，不能再拿它调参并宣称是独立验证。2021的“同配置复现”与“新方法开发”角色互斥，须另签方案，不能在看到结果后切换；2022只有方法与所有选择最终冻结后才另议A8。

本版不承诺T5日期、GPU卡时、投稿时间或运行结果。相关批准全部MISSING。
