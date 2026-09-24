# 实验产物合同

## 1. 通用记录

所有正式stage必须保存machine-readable JSON、完整stdout/stderr、job ID/平台状态/returncode、启动结束UTC、source/环境/资产hash。
最少字段：schema_version、run_id、stage_id、status、evidence_status、source_commit、worktree_diff_sha256、imported_source_files、config_sha256、parent_artifact_sha256、seed、device、dtype、torch/torchvision/lightning/timm/xformers版本、TF32/determinism、inputs、outputs、metrics、failures、costs。
`status`使用OBSERVED/TO_BE_RUN/MISSING/BLOCKED表示证据存在状态；实际判定另用`verdict=PASS/FAIL/INCONCLUSIVE`。未来结果metrics=null，不能填0假装没有损害。

**哈希不是科学效力证明**。必须消费确切父产物，并核对实际加载的字节/张量/配置；一个自述pinned字符串或文件存在不够。
输出写到新目录，原证据只读；临时文件完整验证后原子发布manifest。必要计算失败则不提交PASS；非关键cleanup记录warning。

## 2. S0 bundle

- frozen gate_config.json：ps4、官方commit及实际导入文件hash、checkpoint字节SHA、raw NPZ SHA与有效normalization身份、真实变量序/lat/lon、输入hash/dtype/shape、注册1/4/12步、1e-5容差、执行环境。
- raw input（或可逐字节复原的内容地址）与input_norm.pt；official和bridge对应时效输出；manifest逐文件SHA。
- s0_gate_result.json：每项required criterion、实际值、比较容差、passed/skip原因、s0_gate_pass、verdict_committed；缺项不得真空PASS。
- 四臂trace：arm/model/transform/source/step、相同输入局部与自由rollout分别编号，各算子max/mean差、argmax变量/格点、是否finite。
- job_result.json/logs/env/import paths，source实际文件hash；张量可保留在持久存储，但交付可访问路径+字节hash，不要求将大库复制进本小ZIP。

## 3. Admission / split

保存raw index与解码UTC对应、t-12/t-6/t/+6/+24/+72的实际索引/时间/content SHA、变量/坐标/norm身份、role、issue_id、process_block_id、exposed状态、原候选顺序和替补理由。
消费者逐行核验完整证书及外层PASS，不接受只抽查2行给8行放行。不能通过重写时间字段将index访问结果变成另一份已准入样本。
原始数据完整性检查可读取目标内容来验证finite/hash，但不能据其预报收益筛选，不能将目标数值送入selector。

## 4. Fs

保存固定初始state、每个允许候选checkpoint、共同panel逐issue loss（初始/终态/F0分别）、每次访问训练曲线、梯度/optimizer/update、训练时间角色及源hash、质量规则与冻结时间、全部失败尝试。
资格判据记录quality_pass、numerical_merge_pass、reload_pass、identity_pass分别，不以其中一项替代其他项。
唯一合格Fs保存fs_adapter.pt、fs_merged_backbone.pt、内容state hash和reference_manifest；后续每个worker加载后再次hash。

## 5. Bank / registry

每expert权重文件、训练来源、seed/update、资格面板、非零响应、finite、共同Fs/norm/source、job/执行身份。最后bank.pt独立重载；保存源expert与组装single-k 6/24/72行为差异和zero-edit等价。
registry绑定严格五候选、固定.25/hold4/max_active1、bank与reference内容身份及所有资格证书hash。
profile使用clone或恢复：记录前后tensor hash/模式/梯度状态，不许修改正式权重。

## 6. Cache

每行key=(run_id,issue_id,candidate_id,lead)，保存native预测、truth的独立sealed区域、Fs reference、native loss/gain、状态、身份、耗时/存储路径/hash。
每issue五候选全12步保存6/24/72，不得把缺失预测记0gain。完整性在issue×候选×时效层核验，失败分母/排队/前向次数/GPU进程时间均记录。
原计划developer/cache labels可用于离线oracle，绝不输入serving或fold验证策略拟合。

## 7. Policy / OOF / evaluation

每fold的完整时间support、fit/calibration/validation IDs、scaler/model/HPO拟合来源、prediction_freeze hash和动作在标签揭示前的时间戳。
评分读取冻结动作，再读取验证标签。输出逐issue native24loss、gain vsFs/static、72harm、no-edit比例和cost，保留所有预先定义机会。
paired bootstrap按时间过程/块保留多候选多时效相关性；独立n不是网格或候选数。阈值delta_min、alpha/power/MDE/n/cap单独记录。
Oracle表仅做hindsight上界诊断；不能给policy传实际验证e0/du。

## 8. 结果决定

decision.json含gate有效性、完整估计目标、阈值版本、CI/有效样本/失败/成本、STOP/CONTINUE/PIVOT/INCONCLUSIVE理由；claims.json逐项关联证据级别。
没有holdout正式执行时该域必须MISSING/NOT_STARTED。下一阶段的空文件或计划表不得列为OBSERVED结果。
