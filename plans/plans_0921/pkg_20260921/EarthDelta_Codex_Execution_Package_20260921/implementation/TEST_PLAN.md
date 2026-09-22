# Tests：不删除旧测试，针对真实缺口增加反例

## 当前状态
README报告192 passed、3 skipped；本审查没有在完整当前仓库重跑。旧starter/ResponseKit测试不属于当前git提交的运行证据。tools/check_math.py仅做独立代数例子。

## CPU gate新增
1. MetricSpec：非均匀Q、不同lead/区域、变量scale、mean/sum差异、各端点transform、符号、BF16预测/FP32truth。
2. Selection：finite路径所有box/count/cost/size约束与连续一致；NaN/inf/empty/重复ID拒绝；noedit始终有；ties稳定。
3. Provenance：checkpointbytes变化、变量名交换、intervalkeys交换、grid实际坐标变更、不同bank/metric/cache均拒绝。
4. Time：UTC在不同TZ不变；missing时次/跨年/最终未来不全；scenarioavailability不得当observed；同天气过程各lead组一致。
5. Bridge：真正非零adapter下zeroedit、nonzero→zero→nonzero、异常hookcleanup；梯度经多步回传至coeff而base不更新；若并发/重算unsupported就显式拒绝。
6. S0runner：必需字段任何failure/skip/missing都无法PASS；不自动pip。local-local结果命名不能是officialparity。
7. Finiteoracle：完整candidate枚举、dev选static、samecandidatecommonmask；subsetoracle标partial；registeredfailures不可删除。
8. 4格：exact替换对象正确；预测标签读入路径poison；noeffect/mean/directgain共享输入；calibrated与analyticone不混。
9. feedbackbaseline：零corrector恒等；一次output修改确能影响之后trajectory；未来真值不可读取。
10. composition：additive_response的cross terms与nonlinear mixed term分开；测试真实有限pairs。

## 真实S0（不得被mock替代）
验证一个实际官方checkpoint、原始source权重的SHA、独立upstream正常路径和本地bridge，在相同nativegrid/69vars/norm、6h固定路径上的6/24h预测一致。不同backend不要求bitexact但tol需实测dtype envelope并记录。上游缺依赖则BLOCKED，不能因explicitsoftmax小fixture通过就授予realS0。

## Integration / scientific
数据role、runmanifest、实际planner→edit→预测→评分链、WBX与小数组独立手算、回归与历史兼容都要做。通过软件测试是必要条件，不等于headroom、可预测性或novelty成立。

## 运行记录
JUnit或pytest原日志保存完整passed/skipped/failed数、命令、版本、工作HEAD。测试跳过要附原因。不要为了数字漂亮删除skip或把未run计pass。
