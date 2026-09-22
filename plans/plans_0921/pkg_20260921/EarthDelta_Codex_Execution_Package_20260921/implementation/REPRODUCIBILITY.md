# Reproducibility

1. 优先当前已可用环境。创建独立env可以，但不得更改用户既有torch/xformers/凭据。脚本import不触发网络、pip或训练。
2. 所有外部依赖pin到完整SHA/版本；源码许可、checkpoint/data许可分别核对。原manifest短SHA是检索起点而非可验证完整锁。
3. 冻结run config、MetricSpec、candidate set与真dataindex。时间标准timezone-aware UTC；文件名/数据集名字不证明全年度覆盖。
4. 固定seed和dtype，记录非确定性范围；FD epsilon敏感性与混合方向单独报告。PyTorch JVP占位不是已支持。
5. 先加载官方权重再注入；不在注入后重新initialize_weights。冻结仅指base params，不等于整段forward使用no_grad。
6. 非零bank训练、e0/du训练、calibration、确认分开。bank变化使responsecache失效；共享cache只在内容identity与信息权限相同。
7. 模型初步test结果已用于改方案则转dev/exposed；确认只能用未读role。模型预训练年份也要核对，不仅adapter年份。
8. 真实执行需要预登记GPU/storage/download/forward-call/HPO上限。不从参数数量推断显存、速度、费用。
9. 官方benchmarks是工具：目标单位、时窗、坐标、reference与分母仍需独立核验。
10. 本包不包含当前私有repo完整副本或权重；重现需有权访问当前repo与合法数据。本包的代数测试和preflight工具不是研究代码补丁。
