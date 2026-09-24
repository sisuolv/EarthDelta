# 不得变更的门槛与本轮修订

本包没有执行任何未来实验；未来结果=TO_BE_RUN，缺证=MISSING。以下为新FP05执行合同。

## A. 保留已认证技术合同，禁止重新解释为更宽松的条件
- S0：ps4、独立官方Stormer、已认证Torch/xformers和TF32关闭；全原生网格，6h步长，注册1/4/12步，`max_abs_diff <= 1e-5`。
- 加载前核对原S0 SHA/实际源码/配置/资产；未改变受保护文件时不重复跑S0。漂移则停止并重新界定认证，不更新字符串掩盖漂移。
- 17个FP04源码pin逐一匹配。新增文件可用；不得直接编辑认证核心。需要修改则停止、获批新身份和回归范围。
- Fs必须是v2被选中的唯一certified bundle，不能回到v1失败Fs或每个worker重fit。
- Fs原资格规则仍属历史认证：H32、lr=.0025；Q2 clips=0、per-visit ratio<=1.25、epoch onset factor=1.02；Q3逐样本方向；Q4训练mean<=.99且max<1；Q5训练72h<=1.01；Q6资格24hmean<=1.02。精确完整规则以原`fs_protocol_v2.json`为准，本包不授权再训练或再筛选。
- Fs合并容差原为`max(1e-5,1.5e-3*delta_max_abs)`且rtol=1e-5；不能拿它放宽S0；合并不是要求逐位相等。
- Bank：K4/r4、blocks18–23、singleton a0=rho=.25、max_active1、hold4×6h；no-edit和窗外continuation都为同一Fs。
- Bank E3仍为own-group mean24ratio严格<1；A1–A5注册probe的比较max=0并torch.equal。不可事后放宽/删专家。它们不是OOS安全门。
- 5个候选完整保留；F0是第6条报告背景轨迹，不进oracle/action domain/HPO。
- 真实历史t-12/t-6/t，完整支持到t+72；主native24h、诊断6h、损害guard72h；e=truth-Fs，du=edited-Fs；全端点先double再差分；Q/归约/单位固定。

## B. 对FP05计划的强制修订
1. **2020H2不得标untouched confirm。** X1已训练56个新增H2 issue。[FA-06/E14]
   将`confirm_window`设null、`confirm_access_authorized=false`；2020H2标`EXPOSED_NOT_CLEAN_CONFIRM`。不要自动改到2021、不要自动下载新年。
   未来清洁确认集须另行全历史曝光审查与授权。当前禁用任何confirm读取/评分命令，即使窗口为null也必须拒绝，不能把null解释为无限制。
2. dev仍拟用`[2019-07-01T00Z,2020-01-01T00Z)`，每个完整support必须包含于dev并与所有已曝光support+两侧24h缓冲不相交。窗口名称不是准入。
   历史2019资格support到July4T18Z，因此2019July6T06Z起报仍触边，拒绝；实际最早可用点由UTC坐标机械计算。
3. 台账必须合并所有v1、X1、v2、FP04、旧Round4、调试、筛选的实际访问；失败/未选中的模型不能删除。metadata-only/content-only与梯度/结果曝光分列。
4. `.0034`仅作为Fs-relative工程目标，**不再声称它来自新分布的理论回本线**。若要更改数字，必须在C-J1前新版本及理由，不能看dev结果再调整。
5. 新域是否超过F0由同批`net_vs_F0`直接判断；dynamic是否必要由相对outer-train-selected static的`dynamic_increment`判断。0方向阈值不是最小应用价值。
6. `harm72_margin`、数值重算界限、block/fold和功效计划尚未完全给出，必须在C-J1/正式结果前冻结。
   **不得从Fs训练侧1.01、资格侧1.02或旧shared-F01.05中自动借一个72h dev容忍值。** 缺失时BLOCKED_PROTOCOL_INCOMPLETE。

## C. 逐阶段STOP
- 证书、权重或源hash不符；原始receipt与产出冲突：BLOCKED/INVALID_EVIDENCE，不通过标签补救。
- 新split ledger漏X1、support碰曝光或confirm、旧入口绕过新消费者：不提交C-J1/C-J2。
- C-J1已有golden anchors/双worker/原Fs重复失败：停止；只允许登记的基础设施重试，不以失败结果选更易样本。
- 数值重算容差必须独立预注册；不能把“CPU与GPU浮点归约差”与真实forecast不一致混淆。
- C-J2任何缺issue/candidate/lead、重复、错误身份或非有限：缓存整体不可用于科学估计。基础设施可对同一冻结集合重试；不能仅留下成功子集。
- 同fold标签进入fit/HPO/PCA/scaler或任何confirm读取：INVALID_EVIDENCE，停止。hash校验早退不能替代毒化不变性测试。
- 固定N一次，不按oracle/CI/季节效果扩样或改fold；基础设施修复不改变统计样本。
- 推理/OOF不得读真实候选response/未来truth/oracle。单独的离线评分路径可读全部候选结果，但只能在OOF预测冻结后。
- 未证OOS效用是FP05要回答的问题，不是拒绝FP05的循环前置要求；但FP06不得在该证据前宣布科学成功。

## D. 结果到决策（仅适用于冻结的回顾性H2开发设置）
- `oracle_gain_vs_Fs` CI上界 < .0034：BELOW工程目标，STOP_CURRENT_BANK_FOR_THIS_SETTING；不推断所有bank/季节无价值。
- Oracle有空间，而ridge/regime对Fs无增益或被static解释：STOP_CURRENT_SELECTOR或PIVOT_STATIC。
- ridge-vs-static方向CI跨0：INCONCLUSIVE_DYNAMIC_INCREMENT；不声称需要动态选择。
- gain_vs_Fs超过工程目标，但net_vs_F0未正：不能宣称改善了原骨干，最多说明补偿弱Fs。
- 核心对比通过且72h损害/成本/完整性都合格：仅CONTINUE_BOUNDED_NEXT_ITERATION；不是novelty证明，更不自动解锁confirm。
- CI跨阈值、有效block数不足或功效不足：INCONCLUSIVE；到固定cap停止，不把不显著写成无效。
- FP05只有direct-gain开发，没有response模型对照，任何结果都不能证明双头/response factorization novelty。
