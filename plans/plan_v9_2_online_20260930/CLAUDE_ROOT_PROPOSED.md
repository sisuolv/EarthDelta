# 拟合并的根 CLAUDE.md 内容（提案，尚未生效）

人工审查后再合并到根目录；本文件不是自动授权。

当前拟采纳的计划入口为 `plans/plan_v9_2_online_20260930/00_START_HERE.md`，同时读取该目录 `AUTHORIZATIONS.md`、`PHASE0_PROTOCOL.md`、`STOP_CONDITIONS.md`。不能把v9.1原文或T0未应用补丁当作已采纳规则。

1. 仅执行已签范围；A1含T0范围确认、环境核对、online新增代码和合成CPU测试。依赖/克隆/数据/GPU/后续年份需各自授权。
2. 科学数据仅绑定2020；2019、2021、2022不得自动读取；本阶段不下载。
3. actor/scorer分离；未来监督、失败信息、ID冲突、QC与日志不能影响actor；已发布预报不可回写。
4. 冻结整个 `earthdelta/probe/`、`v8/`、`bridge/`、`static_adapter.py`、`lowrank.py`、`memory.py` 及全部旧 `scripts/probe_*.py`（**包括P5**）。旧probe退出当前范围；T1不提供修改P5的例外。T6需单独A6。
5. 新代码只放 `earthdelta/online/`、`scripts/online_*.py`、`configs/online_*.json`、`tests/test_online_*.py`。记录放唯一runs目录。
6. 从源码检出执行，`PYTHONPATH=.pydeps:.`、`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`、`python -B`。CPU测试需要真实数组/检查点读取守卫，禁用CUDA；跳过不得算通过。
7. GPU由人按A3/A4执行；先拟合期试点，记录分配卡时，最多2H100并行，预算从已签合同读取。
8. 正确面积加权、先池化再开方、F0分母；旧probe面积loss与旧分母不用于新结果。
9. run_id唯一且写一次；新旧配置分开，保存原始逐格矩阵、身份、计时和候选选择过程。
10. 不用 `git add .` / `git add -A`；不自动commit/push，不删 `$OUT`；不覆盖任何历史产物。
11. 工程通过、实验完成、科学支持、新意支持分别报告。Gate由人裁决，INCONCLUSIVE不自动扩样。
12. 每个任务按入口格式汇报；未签事项停在边界，已授权范围内完成具体可审查产物。
