# T2：只实现阶段 0 必需模块（A1）

前置：T1通过、新复用门被采纳、A1已签。本轮没有实现以下文件。约2人日；只用合成CPU，不读天气/检查点，不装arch，不改冻结代码。

## 文件和主要接口

| 新文件 | 接口与必须满足的约束 |
|---|---|
| `earthdelta/online/timeindex.py` | index_of/iso_of/daily_issues；6h对齐、跨年/负数/闰年检查；基于日历而非行号滞后 |
| `clock.py` | SupervisionRecord、FeedbackBundle、SupervisionQueue、Derived；原始监督lineage、artifact版本/可用性；每次release是独立不可变快照，不暴露pending元数据 |
| `stores.py` | ActorStore.read_input(index,t_index) 与 ScorerStore；输入QC当前可见，未来QC只属scorer；actor只收白名单配置视图；阶段0年份仅2020 |
| `qc_reuse.py` | verify_qc_sources、compose_index_manifest；旧缓存身份验证、新17索引/重叠点计划；部分认证不能写全年COMPLETE；合成fixture测试，本任务不读真数据 |
| `losses.py` | area_weights、normalized_step_mse、cells；可微torch归约与NumPy评分分离；明确D是物理MSE；零/非有限分母拒绝 |
| `metrics.py` | relative_rmse_from_mse、weekly_sufficient_stats、circular_indices、paired_intervals、calendar_pairs；所有家族/掩码共用索引、非线性重新归约；NumPy实现 |
| `grad_utils.py` | target_metadata、weight_gradient、response_effect、calibrate_amplitude、unit、cosine、serialize_gradient；不写死小模型尺寸；FP32原梯度+FP64范数；无sketch |
| `dabc.py` | calendar_ewma、DABC；按原始label_available日历年龄的归一化指数权重，不按usable_after/文件生成时刻；query不改变状态、每记录消费一次；ewma_g复用同一归约 |
| `outcorr.py` | fit_eof、make_fresh_features、fit_ridge、select_fit_config、predict_correction；只有Ridge；fresh覆盖所有六变量×四步；每目标同lead历史EWMA固定τ7 |
| `scripts/online_cpu_tests.py` | 真实数组/checkpoint打开前守卫、独立临时目录、准确跳过统计；不绕过冻结文件 |
| `tests/test_online_*.py` | 下述负例与端到端合成场景 |

## 三处容易再次实现错的边界

1. **时钟**：worker可先计算，actor不能看pending列表；连ID重复、异常和日志都不能透露未来。label_available、artifact_available、usable_after分开保存，使用时取依赖最大值、衰减按原始标签年龄。离线拟合/calibration必须和有效在线发布分开入口，不能为校准伪造artifact历史可用时刻。
2. **编辑器**：进入前校验所有block/shape/device/dtype；捕获中途注册失败，仅清理本次hook；禁止共享模型并发；还原requires_grad、eval模式/RNG并对权重buffer做digest。有编辑反传默认关闭checkpoint，安全梯度要与eager和中心有限差分一致。
3. **OCL**：估计期拟合EOF、均值、特征尺度和Ridge；选择期仅选K/正则。缺特征训练行剔除，无零填补，无嵌套CV。最终重拟合只用估计+选择目标，且不在03-31之前发布最终模型的历史预报。

## 必须新增的验收测试

- 日期：首末日、1464/1460边界、精确lag1/lag30、不用最近有效记录替补。
- 队列：源数组和返回快照都无法修改队列内容；未释放ID冲突不改变actor；未来真值NaN/极值/缺失三种投毒下，actor预测、选择、状态、标准化异常与日志一致。
- 缺失：坏输入全臂NO_FORECAST；未来真值只掩对应格；09-27的中间缺步不删除整个窗口；06-26/27、09-30/10-01/02的梯度包/回退按DATA_POLICY。
- 信息公平：比较原始监督ID集合，OCL-fresh包含lag1/lag1c标签来源；只提供单变量或只提供第1步的假实现必须失败。
- 冷启动与隔离：01月初训练缺行计数；03-01可使用02-29已兑现记录；最终重拟合不加入gap目标。
- 统计：不等周有效数下仍等于逐issue pooled RMSE；空周不压缩；所有臂配对；家族成员/数量精确；F0改分母或逐issue比值平均的负例失败。固定种子依赖模拟为诊断，不能测试“所有强依赖下必达95%”。正式覆盖校准另在签前做。
- 面积与数值：常数误差不随W改变；WBX、v8、手写公式同网格一致；真实valid_time/坐标错配拒绝；可微loss与有限差分一致。
- 梯度：双通道校准反例；T=5与冻结T=21函数对账；raw均值与单位均值差异；FP32落盘方向cos>0.99999；零范数显式回退；小幅中心化并不冒称正交。
- 对照：仅慢漂移的合成序列不应被“胜过static”规则误标成短记忆新意；主Gate必须检查ewma_g/OCL-fresh/错位对照。另用有快速可预测创新的fixture检查管线能检出预设信号。
- 快照：原始F0缓存不可回写；同run重复写拒绝；生产rollout小模型注入不绕过真实形状检查。

只在新合成路径中实现这些验收；源代码判断属实不等于已经修复。原测试不能出现无法解释的新失败；必需负例/WBX对照若跳过则T2未验收，不能以可选arch跳过混淆它们。

输入：机器合同、名册规则和旧复用证据。输出：代码diff、新合成JUnit、受限全CPU回归、负例日志、保护文件哈希。运行命令由新online_cpu_tests入口执行；完成后交付，不自动进入未签GPU阶段。
