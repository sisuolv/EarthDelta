# Codex 本迭代状态

Package ID：EarthDelta-R2-NextIteration-20260921。R2-P0-* 与旧 P0-* 不同，不沿用旧勾选。

## P0 — 只做下一迭代

- [ ] R2-P0-01 最小可信执行路径：官方S0/身份/Q/样本入口。状态：READY；授权仅源码+合成CPU。
- [ ] R2-P0-02 合格小bank、完整候选缓存、oracle/static gap。状态：LOCKED。
- [ ] R2-P0-03 合法policy、四象限、direct-gain/regime、标签效率。状态：LOCKED。
- [ ] R2-P0-04 有信号后才做actual/virtual/output/feedback挑战。状态：LOCKED。
- [ ] R2-P0-05 只读汇总与继续/停止判决。状态：LOCKED。

Current task: R2-P0-01  
Do not proceed beyond: R2-P0-01  
Automatic unlock: false

## Unlock condition

R2-P0-02需要：R2-P0-01真实同配置独立S0 PASS + 所需小数据子集/模型证书 + reviewer明确授权 + 非空资源cap。CPU通过不能替代真实S0。

R2-P0-03需要：E1的正式HEADROOM_PASS，或reviewer明确批准的DEV_PROMISING；必须保留confirm封闭且有剩余策略预算。

R2-P0-04需要：E2出现可信合法信号，并为纠错挑战单独批准资源；dual无增量时只能以缩窄claim继续。

R2-P0-05仅在单独授权后只读汇总，可汇总早停分支；未到达的任务不是成功跳过。

## 每次完成/阻塞时

在工作副本中记录code_status、run_status、research_verdict和artifact路径/hash。只有所有成功条件实际满足才勾选任务。CPU部分完成/真实S0缺失时保持未勾选，记录BLOCKED。

停止并返回本次修改、实际测试通过/跳过/失败、缺项、下一任务建议。不要因为保存了decision.json或依赖PASS就自动更改批准任务列表。原始交付包的checksum不随工作状态更新；在单独工作副本维护状态。

## 当前禁止

不训练bank，不运行GPU/checkpoint加载/真实天气实验，不访问confirm，不扩展P1/P2，不自动下载，不push，不重写无关模块。
