# 四层测试与结果验证

## 既有证据，不重复冒充本轮运行

Fs v2 prereg log：766 passed/7 skipped。Bank refreeze log：861 passed/7 skipped。
已读`test_plan_gate_chain`/`test_loaded_admission_binding`/`test_profile_immutability`的断言，支持对应子范围。
本包没有重跑这些测试，也没有GPU。新CPU结果必须另存log/XML/argv/source，不能复制历史数字。

## A. CPU实现/单元反例（FP05b）

1. 曝光账本：X1-N64的56个H2训练issue必须入账；旧v1失败、探索override、资格筛选亦入账。仅凭“最终Fs未用”不能删除。
2. 2019H2窗口边界：完整support触碰2019资格曝光+24h则拒绝；确认null必须拒绝所有confirm。
3. 候选替换：只按固定时间/完整性规则，tie/重复/越界/替换次数按冻结规则；不能依据native loss替换。
4. 所有入口：admission、launch、worker读取、merge、fit、inner-HPO、变换、freeze、score；绕过单独helper的旧入口也必须在新wrapper被拒绝。
5. policy_dev不伪装bank_fit；不是mock掉证书检查后声称绑定通过。真实loader测试先成功，再各改一个字段应拒绝。
6. 实际weighted/native损失端点先float64差分；负/NaN权重、零scale、空objective拒绝；F0不进入5候选。
7. cache N*5*3覆盖：缺行、重复、错ID、wrong bank、old source、非有限、失败重试未计费都拒绝。
8. features只依赖合法历史与calendar。真值/实际candidate forecast毒化不改变feature内容；测试fixture哈希需自洽，不能只走checksum早退。
9. nested OOF：scaler/PCA/kmeans/lambda从合法train内拟合；fold F标签毒化时F的预测不变，而其它fold变化不构成失败。
10. 同一issue候选不跨fold；相邻完整support净化；不把issue_id不同当support不重叠。
11. 没有预测冻结文件或freeze被改则评分失败；重复run不覆盖旧预测。
12. runner未授权/前驱FAIL/BLOCKED/MISSING时，从调用计数验证下游函数未被调用。

## B. GPU debug（FP05c）

- 不重跑已认证训练。加载Fs/bank，用8个已曝光anchor校验；只对已有历史单元格声明golden一致。
- 所有5个candidate + F0背景；两worker同输入，输出逐位比较；同模型串行，不共享hook并发。
- 同一Fs/no-edit、hold后Fs继续、source17pin和实际weights保持不变。
- 6/24/72从一条12步轨迹提取；primary/tolerance不改；独立CPU归约使用预冻结的数值界限。
- 只测资源/完整性；这些结果不进入效果表、不选择dev窗口或改变阈值。

## C. 回归与artifact测试

原有S0/Fs/bank核心源若未改，仅验证hash/certificate。新增薄包装的坏certificate/错weights/错split测试必须失败。
若需要修改17pin，暂停并登记重新认证，而不是只重跑几项CPU后沿用原GPU证书。
profile/horizon使用deepcopy；新增测试以(block,name)为参数键，避免不同block同名覆盖；hash与梯度flag均核对。

## D. 科学结果验证（FP05f/FP06）

- 全样本共同分母，oracle!=legal；best-static OOF选择，regime次要。
- 独立重新计算native loss/gain和各estimand；避免mean ratio和ratio of means混用。
- paired time blocks保留全部candidate/lead/method，不能当网格点或seed为独立样本。
- CI与工程阈值/MDE分开；N冻结一次，跨阈值就INCONCLUSIVE，不再扩样。
- 72h harm阈值与成本合同必须结果前固定；仅point estimate未越门不能称统计无害。
- 所有科学正面结论限于回顾性H2开发；FP05并未检验response模型novelty。
