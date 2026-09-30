# v9.2 复用决定（替代原补丁的执行建议，不改旧审计）

| 组件 | 方式与来源 | 新边界/验收 |
|---|---|---|
| 模型、生产rollout | import `probe.rollout.load_bridge/rollout_trajectory` | 固定69×128×256；合成测试注入小模型；不声称此函数本身通用 |
| 全矩阵编辑 | import `DirectWeightEditor`，新调用封装 | 全量进入预检；只清理本次拥有的hook；失败/并发/模式/RNG/冻结状态负例 |
| LoRA（未来T5） | import `build_fs_adapter/ExpertLoRA/controlled_rollout` | 共享backbone串行；不包外部checkpoint；seed初始化语义按实际源码 |
| 评分 | import `v8.standard_metrics.issue_mse/relative_rmse`、WBX | 小形状薄封装、time/valid/坐标身份绑定、共同掩码与F0分母 |
| 可微损失 | 新写简短torch面积归约 | 归一化完整H×W；与v8/WBX/手写对账；不强行复用布局耦合的area_weight_q |
| 统计 | 新写NumPy循环周块与分位区间 | 采样器可很短，但掩码/非线性统计/覆盖校准/MDE的验收不能省；不承诺总共40行 |
| OCL | 已有sklearn `Ridge`+NumPy SVD | 单一估计/选择划分，不嵌套CV；fresh特征完整，选择后重拟合严格按目标行角色 |
| 记录/时钟 | 参考memory时间规则，新队列/Derived | 不向actor传pending列表；不可变快照、防异常/日志泄漏 |
| 衰减 | 新日历EWMA；memory仅作每日一条特例对账 | DABC和ewma_g使用同τ定义、真实日期年龄；不以记录条数代替天数 |
| QC | 复用resume01的绑定缓存/逐索引JSON | 身份核验、17项补查、8重叠对账；不把抽样当当前store全年认证 |
| 效应比 | 参考顶层 `calibration_effect` | 69通道能量公式、T≥5、B=1逐issue，冻结21状态函数仅作合成对账 |
| 相关工作 | PETSA/ORCA/INC只读；batch5延后 | 不拷NC/SA或无license代码；TAFAS/OnlineTSF证据仍MISSING |

内部源码与第三方license的证据沿用旧T0逐项表。没有新增vendor代码。旧480行节省估计依赖arch方案，不原样套到本版新增的统计/信息边界；后续按实际diff记录代码量，不用行数目标削弱验收。

目前已有依赖满足主线：NumPy1.26.4（`.pydeps`实际模块）、SciPy1.12、sklearn1.2、WBX路径下CPU JAX0.6.2。保留版本，不根据重复dist-info误判成NumPy2.2.6；冻结前再次记录实际导入路径。缺依赖时区分“现有必需接口缺失”与“可选arch未签”。
