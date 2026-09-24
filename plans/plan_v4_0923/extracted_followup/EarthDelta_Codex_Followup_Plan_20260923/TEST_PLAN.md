# 测试分层：CPU不能替代GPU，GPU不能替代科学结果

## A. 当前证据

OBSERVED：Round4 targeted307、additional127原始日志已复制并核对git blob身份；它们不是本次重跑。最新66/3仅摘要，原日志MISSING。当前仓库pytest/GPU/训练本次均未执行。

## B. CPU correctness（TO_BE_RUN）

保留所有旧测试；新增测试必须调用真实消费入口，不只测一个可选helper。

1. `test_input_inverse_source_precision.py`：真实或独立FP64输入std源，先inverse后cast vs先cast；固定相同tensor仅policy不同影响身份；input与diff路径均覆盖；不能mock掉实际normalization。
2. `test_plan_gate_chain.py`：真实stage dispatch在前驱False/exception/MISSING时trainer/cacher spy调用数0；单阶段入口同样拒绝；发布前关键异常不保留PASS。
3. `test_loaded_admission_binding.py`：外层FAIL、漏一行证书、6h证书供24/72目标、错UTC但index合法、错norm/source/坐标、少t-12、角色support交叉必拒绝。
4. `test_metric_contract.py`：端点先double；每目标Q denominator，batch/candidate维不被吞；negative/NaN/zero slice/scale0拒绝；legacy路径不许替代主目标；analytic与calibrated分开。
5. `test_fs_quality_gate.py`：用保存的有限但明显变差面板构造质量FAIL，不用nonfinite代替“训练退化”；共同初始/最终checkpoint比较；质量失败时freeze/bank未调用；少updates不是成功证据。
6. `test_profile_immutability.py`：真正profile optimizer路径作用于clone；正式state/buffers/mode前后相同。
7. `test_fitted_fs_bank_roundtrip.py`：跨进程保存/加载；不同Fs/source/norm拒绝；动态tensor变一字节/改mapping拒绝；组装singleton正确索引；reference/continuation不能切F0。
8. `test_plan_cache_evaluation.py`：缺candidate/重复行/错身份/NaN失败；验证truth或实际candidate response读取触发权限/loader异常；scaler仅foldfit；oracle不进policy；所有动作同issue同role。

还要验证未启用模式明确拒绝/隔离：共享模型普通forward+controlled混用不在本run允许范围；禁止activation checkpoint。

## C. 真实GPU gate（TO_BE_RUN；BLOCKED直到资产/授权）

S0四臂与独立official export/gate全部1/4/12步<=1e-5，不换容差和变量。
LoRA短步测试记录A/B真实梯度、backbone冻结、hook清理、显存、吞吐；B零初始化导致第一步A梯度零不是自动bug，需后续B变化后验证。
Fs质量与merge、bank qualification/assembly是独立GPU门，不由S0代替。

## D. 科学证据（TO_BE_RUN；所有前驱PASS才可运行）

冻结dev矩阵→时间purged OOF预测先冻结→native评分→paired过程bootstrap。
独立复算oracle vsFs/static和合法policy vsstatic。检查n/分母/全部失败/成本；delta_min和MDE分开。
没有合格完整bank不能下“策略没有价值”；hindsight有余量不能下“policy有效”。

## E. 交付工具测试

本ZIP的schema/manifest/DAG/状态验证只证明交付结构。验证报告会明确`repository_tests_run=false`与`gpu_runs=0`。绝不把交付检查通过数量写入EarthDelta算法测试分母。
