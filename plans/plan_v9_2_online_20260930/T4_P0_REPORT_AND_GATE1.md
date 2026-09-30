# T4：复算、分层结论与人工 Gate 1

新脚本 `scripts/online_report.py`、`online_calibrate_statistics.py`、`online_power.py`；CPU测试用完整合成run，不读新天气数组。统计校准/功效工具在拟合后、分析前使用；报告在真实分析后使用。A1负责开发，数值输入必须属于已签A3/A4产物。

## 分析前收据

输出COVERAGE_CALIBRATION.json（全部场景/家族、c的选择、独立验证、不确定性）、MDE.json（逐比较及交集功效）、BUDGET_PROJECTION.json。没有实际运行就写MISSING，不写空表充当验收。`online_calibrate_statistics.py --phase fit --contract <...>`只接受fit来源；输入含analysis角色立即拒绝。

## 主报告

1. G-0：HEAD/diff、授权、数值环境、20步/梯度门、数据组合manifest、因果负例、覆盖资格、分母/掩码、发布/回退/失败数量。
2. 全臂6变量×4时效物理RMSE和相对F0改善；主格边际95%及家族区间并列。
3. F_B/F_C/F_O/F_M差值、F_H非劣全表；所有差值F0分母、单位pp；静态/慢漂移/新鲜反馈三个问题分开。
4. 比较族、周锚点、空周/不等有效数、14天主结果、7/28天敏感性、校准c及MC限制。报告不能把名义覆盖当真实覆盖。
5. fit选择全表：候选数不同如实列；主臂预先固定；没有分析期选择。既往静态/2019数字只作背景，不冒充2020基线。
6. 诊断：原始/中心化日历相关、OCL-fresh R²、分季节、truth_one_step/lag2/buf12；不把diag当可部署收益或全局上界。
7. WBX主格交叉核对：数组坐标、init/valid时间、channel/std身份显式绑定；不能只凭传入2020字符串通过年份门。
8. 计时与全部实际卡时，预算剩余；ENGINEERING_PASS、EXPERIMENT_EXECUTED、SCIENTIFIC_SUPPORT、NOVELTY_SUPPORT逐项独立给证据与限制。

## 机械状态与路线建议

只读已签机器合同中的阈值，不再硬编码第二套。家族大小为16/24/12/12/4，校验每成员出现一次；不缺报不利格子。区间阈值状态按协议三态；任何QC/因果/求导故障先BLOCKED，不能得到科学STOP。

| 条件 | 报告建议 |
|---|---|
| 数据/工程门失败 | BLOCKED，先修复，不判科学假设 |
| 覆盖资格失败或模型假设不适用 | STATISTICAL_INCONCLUSIVE，不做正式GO |
| G-B/G-C/G-H全PASS，稳健性无反向证据 | 建议有限参数T5；必须人工Gate1与A5，不能自动执行 |
| 输出臂对F0有可靠增量而参数未优于它 | 优先输出订正/重拟合研究；不是已获业务部署许可 |
| 区间跨门或过宽 | INCONCLUSIVE；唯一补证为另签2021同配置，不改2020或阈值 |
| 注册比较的上界仍不达资源门 | 该受限方法在此协议下NOT_MET；不是所有参数编辑不可能 |

不要说“只剩漂移”，除非相关增量区间足够窄且满足预定实用等效标准；原方案没有单凭不显著作此判断的许可。原始lag1−ewma_g与同样中心化的lag1c−delay30分别解释，不能混用。

测试必须包括：一个门未过不得GO；上界明确低于阈值和跨阈值的区别；缺签只能PROVISIONAL；没有arch仍可跑；遗漏OCL-fresh/ewma_g或改变家族成员拒绝；失败slot不能从分母消失。每个数字都能由已保存cells/选择表复算。

最终输出GATE1_REPORT.md、逐格CSV/JSON和复算命令，再等待人判定。未来命令示例：`python -B scripts/online_report.py --run <frozen_run> --contract <signed_contract>`，使用固定环境；当前没有可运行实现。
