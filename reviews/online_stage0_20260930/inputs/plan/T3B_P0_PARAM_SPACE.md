# T3B：固定方向的参数空间增量

前置T3A-fit及T2，拟合期A3、分析期A4分别授权。本任务为冻结F0上的单次编辑，不训练长期online状态，不能证明胜过完整Fs*或滑窗重拟合。

拟新增脚本 `online_p0b_gradworker.py`、`online_p0b_edits.py` 与 `tests/test_online_p0b.py`；复用桥接/编辑器，在线安全封装来自T2。

## 梯度与身份

对每个有合法输入和完整四步监督的issue，用正确的六变量×步骤1–4归一化MSE求原始梯度。D_est_short只用estimate F0正确MSE，初始模型/分母/状态归一化/层集合全部绑定。blocks默认18–23的attn.proj.weight，维度从模型读取，不在小模型测试中写死6291456。

原始梯度以FP32保存；范数FP64，均值FP64累计。往返方向cos>0.99999；记录min/max、范数、若转fp16的下溢比例（只作诊断）、真实dtype与校验哈希。零梯度不算异常：对应单位方向为空，回退F0并计数；非有限梯度技术BLOCKED。未来梯度的范数和失败状态同样不能泄漏。

冻结均值 `g_mean_est=mean(raw_g_s)` 只用estimate合格目标行；不平均单位梯度，不在最终fit重算。所有标量/向量归约约定写进config_resolved。

## 五个方向族

| 族 | 定义 | 地位 |
|---|---|---|
| lag1c | −unit(g[t−1日]−g_mean_est) | 预先固定主臂；不保证正交 |
| lag1 | −unit(g[t−1日]) | 次要未中心化对照，F_M与raw ewma_g比较 |
| static | −unit(g_mean_est) | 固定方向基线 |
| ewma_g | −unit(已释放raw梯度的日历EWMA) | τ网格同DABC，慢漂移基线 |
| delay30 | −unit(g[t−30日]−g_mean_est) | 与主臂同样中心化，只改变日历年龄；替代random |

delay30特意同样中心化，避免把“raw旧梯度 vs centered新梯度”误叫纯时间对照。raw lag1−raw ewma_g单列为F_M，不需要新GPU臂。缺精确日期时回退F0，禁止从幸存记录中拿最近一个。

## 拟合、校准与冻结

estimate阶段离线计算均值/分母；校准16日为02-02..02-17，各族及每个ewma τ都用同一日历集合。静态拟合artifact可使用整个estimate，这是训练而非这些历史时刻的可部署结果；滞后反馈本身仍只取对应截止之前的数据。不得混进select/analysis标签来改均值或幅度。

8个校准配置=lag1c、lag1、static、delay30各1个+ewma_g四个τ；均以24h全69状态标准化能量比中位数0.1为目标，括区/二分/失败规则取机器合同。选择期各配置3个倍数，共24组；ewma从τ×倍数12组中选一组，其他族各3组。候选/失败/并列全保存。没有在lag1和lag1c之间再选主臂。

分析前均值、幅度、倍数、τ、OCL/输出基线及主臂已经完全冻结。分析每个有效issue五条20步rollout，F0取共享库；不能在看到T3A分析效果后削弱或加强某臂。

## 诊断约束

39个固定日期：truth_one_step、lag2、两条buf12，共4条20步预报。truth_one_step用当期四步梯度，是不可部署的一步试探，不是上界；lag2复用已有梯度。buf12使用前一天的两步目标，两种缓冲在每日00Z应有相同可用记录，比较预测/状态/异常日志。仅增加两步梯度计算。

eqnorm、多倍数全表、random及sketch不执行。完整梯度滞后1–10日余弦只作CPU机制诊断，不能拿相关性替代Gate。需要增加诊断须签前改版本/预算。

## 输出、验收与停止

输出：每issue梯度manifest、raw FP32与FP64范数、选择全表、每候选校准轨迹、config_resolved、逐臂cells/掩码/回退计数、lineage、计时、实际卡时。单步梯度前向已经计入预算，不重复将无梯度缓存当可微结果。

负例：原始fp16小梯度下溢；raw均值与单位均值差异；非同中心化的delay对照拒绝；未兑现梯度数量/异常泄漏；第二个block失败留下hook；NaN梯度不能被掩去；fit外标签用于均值/幅度必须失败。

未来命令：`online_p0b_gradworker.py --config <...> --phase fit|warmup,analysis` 与 `online_p0b_edits.py --config <...> --phase fit|analysis`，从repo根以`.pydeps:.`/protobuf环境执行。脚本当前不存在。A4前不提交analysis worker，即使worker是离线也不允许提前访问分析期监督做选择。

校准/求导/身份/因果失败为技术BLOCKED；零范数/缺合法日历记录按预定回退；统计结论由T4产生。所有费用共用BUDGET中A3≤3、A4≤5、总≤8的待签上限。
