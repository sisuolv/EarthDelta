# 阅读与核查边界

状态：OBSERVED。审查日期：2026-09-23。

- 实际分支头 `fdd92d79b1b03e8397d4c487d7fa4a347ae6fc37`；代码实现 `a3596e6b804e9d23b72d1247b08c47129d6b50b1`；比较基线 `d749a1c62521226df857587e08f7d067b0f15355`。
- 两个HEAD的earthdelta/scripts/tests等代码树相同，plans改变。不是最新代码已修复。
- 已完整阅读附件source_plan全部五文件、状态/索引、三份reference_reports；先计划，后报告，再当前代码与日志。
- 已读取目标提交diff中的相关改动和当前代码关键范围，不宣称逐行阅读整个巨型diff或整个仓库。
- 日志307/127确实取回；其原始文本在observed_logs，复制后的git blob SHA与GitHub相符。
- 最新66/3原始测试日志、616 issue完整admission、pt-7ant09uv原始结果/日志/张量未取得。
- 历史GPU执行依据gpu_records_recomputed中嵌入的job_result和run记录；不等于本次重跑或大checkpoint独立重新hash。
- 无新文献综述；novelty/weather utility判断只回答是否有合格实验支持。
- OUTPUT_SCHEMA.json是字段/枚举模板，不是带type/properties的正式JSON Schema。本包验证器按其字段、类型与枚举核验，不伪称运行了不存在的正式schema。
