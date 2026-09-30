# 复用地图：实际基准 d55ad70

本版使用 `probe-headroom-20260927@d55ad70`。完整66符号签名和源码哈希见旧T0 `SYMBOLS.json` / `SYMBOL_REVIEW.md`，本轮已验证其来源哈希；具体取舍见本目录 `REUSE_DECISIONS.md`。

| 文件/符号 | 需要记住的实际语义 |
|---|---|
| `probe/rollout.py::load_bridge,rollout_trajectory` | 固定生产尺寸；第0状态为输入；20步为21状态；梯度前向不能由磁盘预测场替代 |
| `probe/rollout.py::area_weighted_mse` | 面积MSE多W因子；不得与新分母混用；冻结，不修旧代码 |
| `probe/edits.py::DirectWeightEditor` | 部分__enter__失败会遗留hook；上下文退出不会自动拯救失败的__enter__ |
| `memory.py::VerifiedRecord,eligible_records` | frozen dataclass不冻结tensor；pending ID重复可先于过滤抛异常；不可作安全边界 |
| `static_adapter.py::build_fs_adapter` | 指定seed会覆盖A为std=.02正态；B默认0时初始零编辑 |
| `bridge/stormer_bridge.py::controlled_rollout` | 拒绝共享backbone重入；不能识别任意外部checkpoint包装 |
| `scripts/probe_p3_directions.py::calibration_effect` | 21状态断言但只读第4步；全部状态标准化通道先合并能量再求比 |
| 同脚本target_params/flat_delta/effect/calibrate | 前两项block范围写死；后两项是main局部函数，不能直接import成通用校准器 |
| `v8/standard_metrics.py::issue_mse,relative_rmse` | NumPy评分，不可直接反传；单变量[issue,lead,H,W] |
| `wbx/evaluate.py::evaluate_single_chunk` | 调用者时间门不等于与数组坐标绑定；需要显式包装检查 |
| `probe/contracts.py::issue_index` | 缺跨年/负数上界检查；online时间模块补齐 |
| `v8/data_repair.py` | 不作为年份授权边界，也不能用部分索引成功代表全年完整 |

六变量通道索引、物理单位和时效统一取机器合同；实现启动时与DEFAULT_VARIABLES对账，不靠复制数字默认为正确。状态mean/std与差分mean/std必须区分。

只支持源码检出，脚本从repo根目录运行。`scripts`可作namespace导入但wheel不会打包它；尽量import库级函数，局部函数的新通用封装只放online模块。命令使用`.pydeps:.`和protobuf环境；禁止自动安装依赖来掩盖版本问题。
