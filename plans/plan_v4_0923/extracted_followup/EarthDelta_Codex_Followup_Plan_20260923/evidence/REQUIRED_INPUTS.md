# Codex启动前与每阶段必须补齐的输入

## 可以立即做

本包已包含原计划五文件、三份reference_reports、schema、审查JSON及两份历史pytest日志。阅读和交付校验无需额外上传。
CPU源码补丁需要可访问的真实repo；若Codex已有checkout，不必再次上传全repo，但必须记录版本/未提交diff。

## FP-00/FP-01必需

1. 最新实际checkout或受审source snapshot（earthdelta/scripts/tests/reference源码），含a359及以后diff、环境overlay/source实际导入路径；旧base d749差异用于追踪。
2. **pt-7ant09uv** 原始`job_result.json`、完整`stdout/stderr`、`rerun_stages.json`、`s0_gate_result.json`和全部criteria、实际`gate_config.json`；摘要无法替代。
3. 对应official参考bundle：manifest、input_norm、raw input或内容地址、6h_1/4/12step outputs、坐标/变量、各文件SHA。最新完整raw路径在本次材料中未给出，请从该job manifest解析，不猜目录。
4. 同运行的checkpoint字节、实际normalization NPZ、lat/lon/channel输入、对应environment与xformers算子probe日志。可放在Codex可读取的本地/持久路径，无须上传几GB权重到聊天；hash与身份必须可核验。
5. S0-only资源授权和cap（当前未知），RUN/ASSETS/OVERLAY_RUN真实路径。没有这些只做CPU诊断，不能提交expert jobs。
6. 若要把“66 passed,3 skipped”纳入本次证书，补原始pytest log/XML、选择表达式、source hash/环境。307/127历史日志已在包内，不要重复当最新测试。

## FP-03之前补充的历史Fs取证

- `artifacts/round2_cci/ed-r3-j4-bank-formal500-0922004700-0a3c68/` （job pt-3x63g0c6）及 `artifacts/round2_cci/ed-r3-j4-bank-formal-v2-0922011307-2d8e12/`（job pt-cdj1s2le）的四worker原始run_record/fs_fit_record/fs_post_freeze/stdout、job_result与实际argv/source快照。
- 各worker的fs_adapter与fs_merged_backbone（可访问路径+字节/state hash），共同初始状态、losses/training_issue_ids、panel原始loss。记录不是大权重重新hash的替代。
- 旧admission原始文件和content certificates（报告指8行/2证书/6h准入）；用于复现消费漏洞，不可直接拿它训练。

## FP-03/04运行新实验前

- 真实Zarr与time/channel/lat/lon，原计划t-12/t-6/t/+6/+24/+72全部内容证书；冻结bank_fit/policy_dev/exposed IDs与support区间。
- 已批准训练侧Fs/专家质量规则、重试/LR/更新cap、资源profile；未定义字段保持TO_BE_PREREGISTERED并阻止正式启动。
- 只有真实合格Fs产生后才能补Fs certificate；只有合格bank产生后才能补bank/registry。不能要求用户上传“未来已成功结果”以假装现在完成。

## 后续（现在不要求上传）

完整开发cache、OOF冻结预测、native评分/成本/失败和decision均TO_BE_RUN。confirm/dual-head未解锁，不要求现在准备它们。

不要上传账号token/API key/SSH私钥。仅需要授权工具或本地可访问的路径与非敏感配置。
