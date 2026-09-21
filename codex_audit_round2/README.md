# EarthDelta 第二轮审计与独立研究复核

本目录汇集第二轮审计、CPU 证据和供 ChatGPT Pro 独立分析的材料。研究判断与实现准入是两个独立问题；这里的报告本身也需要接受质疑。

## 当前结论

| 维度 | 第二轮判定 | 含义 |
| --- | --- | --- |
| Novelty / 研究价值 | `PROCEED_TO_P0_04`；无人占据判断 `UNCERTAIN` | 只支持有 cap 的最小假设检验，尚未证明新颖性或实际价值。 |
| 代码准入 | `FAIL_REOPEN_P0_03`，同时补 P0-02 | 15 条 finding：2 P0、10 P1、3 P2；当前不解锁模型训练。 |
| CPU 测试 | 283 passed / 7 skipped / 0 failed | 是实际复跑结果，不代表真实 upstream S0 通过。 |
| GPU / xformers S0 | `BLOCKED`，未执行 | 已知资源限制；不是本轮新增实现 finding。 |

受审范围是 `4fe55a7..fb767f7`，受审 HEAD 完整值为 `fb767f7f6efbc428be39c9ad84f5905331d6e40f`。本目录的交付提交发生在受审 HEAD 之后，只整理审计材料，未修复报告中的问题。上传分支也包含原先尚未推送的 `607fad5`、`fb767f7`，以便核对被审代码。

## 给 ChatGPT Pro 使用

1. 上传 [CHATGPT_PRO_REVIEW_PACKET.md](delivery/CHATGPT_PRO_REVIEW_PACKET.md)。它是一份独立可读的 Markdown，包含研究规格、第一轮相关材料、第二轮结论、CPU 证据及逐 finding 源码片段。
2. 将 [CHATGPT_PRO_PROMPT.md](CHATGPT_PRO_PROMPT.md) 全文作为提问。它要求优先独立分析 idea，再审视审计结论，明确支持或推翻哪些论点。
3. 需要完整源码上下文时，另提供 [完整 ZIP](delivery/EarthDelta_Round2_Pro_Review_20260921.zip)，或解压后提供所需文件。ZIP 带完整受审模块/测试快照、官方 Stormer 相关源码与 MIT license。

仓库当前是私有仓库，不应假设另一会话能直接读取 GitHub 链接；附件是完整信息入口。若该会话不能解压 ZIP，先使用 Markdown packet。不要以“已提供链接”代替实际读取证据。

## 阅读导航

- [原始背景 AUDIT_BRIEF.md](AUDIT_BRIEF.md) 与 [原始审计任务 AUDIT_PROMPT.md](AUDIT_PROMPT.md)：保留请求原貌，其中的内部自述不是独立事实。
- [完整第二轮报告](results/ROUND2_REPORT.md)：先研究判断，再六项 disposition、15 条 finding 与验证范围。
- [研究判断 JSON](results/novelty_value_assessment.json)、[findings JSON](results/audit_findings.json)、[代码判定 JSON](results/round2_summary.json)：独立机读结论。
- [CPU 测试日志](results/evidence/pytest_cpu.log)、[CPU 反例](results/evidence/cpu_repros.json)、[合同反例](results/evidence/contract_repros.json)：实际证据。
- [测试改动核查](results/evidence/test_change_review.json)、[实际 diff](results/evidence/audited.diff)：修复与测试是否真正落实。
- [第一轮材料与上游源码来源清单](context/SOURCE_MANIFEST.json)：原路径、commit、SHA-256 和副本路径。
- [交付文件校验清单](delivery/DELIVERY_MANIFEST.json)：ZIP 与 Markdown 的 SHA-256，以及 ZIP 内每个文件的内容哈希。

## 材料边界

`context/round1/` 是原第一轮包的选定文件，按字节复制并记录来源；其结论不是本次新增事实。`context/upstream_stormer/` 来自官方 Stormer pin `58dfee5a6037399a40fefd492bc00421e0c885a8`，保留 MIT license。报告中的官方源码链接指向这一独立副本。

`results/evidence/final_integrity.json` 证明的是审计结束时受审源码未变，并不要求交付后的 git HEAD 仍等于受审 HEAD。本次整理只调整报告中的源码链接，使 GitHub 可阅读；finding 内容、测试结果和原始 diff 保留。

交付包不包含模型 checkpoint、天气数据、后台拉取产物或完整第三方仓库。CPU 复现脚本留作证据；包本身不是完整可运行环境，尤其不附带归一化二进制数组和真实 S0 输入。

## 重新生成交付包

在仓库根目录运行：

```bash
python3 codex_audit_round2/build_pro_bundle.py
```

生成器从固定受审 commit 取源码，从本目录取审计与背景材料；ZIP 使用固定时间戳和排序，避免当前工作区代码漂移改变被审对象。生成器不运行模型、不拉取数据、不调用 API。
