# T1：环境核对与旧探针退役说明（A1）

本任务不修P5，也不尝试恢复P4/Step2。只修P5不能生成合规headroom结果；需要恢复时另立任务、精确解除冻结、重新审查整链。当前方案没有恢复预算，也不把未运行探针列进论文证据。

## 操作

1. 记录HEAD、分支、status、diff哈希、实际Python/torch/NumPy导入路径、WBX vendor身份、pip freeze摘要；不要把main当当前分支。
2. 核对根CLAUDE合并、A1以及本版机器合同是否已被人工采纳。缺签不改代码。
3. 在新的 `runs/online/t1_<UTC>/` 运行受数据守卫保护的CPU回归。可将旧T0守卫源码复制到新run的tools，运行目录必须保留同样的repo/runs/online层级；输出/cache/tmp全指向新run，不能原样运行旧守卫覆盖旧JUnit。
4. 报告结果与旧975/62/4逐项差异；已有失败不在本轮范围内则BLOCKED，不放行科学运行。

底层pytest参数固定为：

```bash
PYTHONPATH=.pydeps:. PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
python -B -m pytest tests/ -q -m "not slow" -p no:cacheprovider
```

**不能无守卫直接执行上式。** 由新run中的守卫通过pytest.main传入相同参数，并设置 `PYTHONDONTWRITEBYTECODE=1`、`CUDA_VISIBLE_DEVICES=`、`JAX_PLATFORMS=cpu`、独立临时/缓存目录。未来统一入口拟为 `scripts/online_cpu_tests.py -- tests/ -q -m "not slow" -p no:cacheprovider`；它目前尚未实现。

不加载任何真实数组、归一化常数或checkpoint；必要的合成数据都写新run。需要真实数据才能运行的用例明确跳过；这叫“受限CPU回归”，不叫所有测试通过。

输入：当前代码、旧T0源码/环境记录。输出：`env_baseline.json`、JUnit、跳过清单、命令日志、保护文件哈希。CPU时间盒半天；GPU卡时0。验收：源码未修改、根规则与授权明确、没有新未知回归失败。此结果只支持工程基线。
