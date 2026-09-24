# 测试分层：CPU、GPU gate、回归、科学证据不得互换

## A. 历史回归（OBSERVED，不重新标TO_BE_RUN）
Fs-v2 prereg日志：766 passed / 7 skipped / exit0。FP04 prereg-refreeze日志：861 passed / 7 skipped / exit0。它们属于当时source snapshot，不是本次审查执行的pytest。
`test_plan_gate_chain.py`有真实dispatcher spy；`test_profile_immutability.py`包含直接profile会改变bank的辨别性对照。继续保留这些测试，不要求为了新计划重训历史模型。

## B. FP05b新增CPU反例
| 边界 | 必须实际构造的反例／断言 |
|---|---|
| Exposure | 加入X1未选模型的H2训练；完整support触碰曝光buffer或confirm就拒绝。不能只看issue日期或最终Fs来源 |
| UTC | 形状相同但时间轴移位、missing interior timestep、邻近窗口只差一个端点，都在模型调用前拒绝 |
| Membership | policy_dev标签但不在冻结id集合；旧admission证书复用于新协议；重复/越界/role-swapped issue，拒绝 |
| Metadata-only bypass | 填`PASS`、伪造省略expected、只重算行hash不对应源row，都不能通过消费链 |
| Artifact | Fs或bank任一byte变、错误norm、候选coeff/window/Q变、SHA/state digest类型混用、证书缺一个step，拒绝 |
| Candidate coverage | N×5全矩阵，no-edit必有，F0不可变第6candidate；漏/重复/多余candidate拒绝 |
| Debug anchors | 只对有旧FP04 own-group记录的Fs/所属expert cell做历史锚定；无对应旧cell不能捏造真值 |
| Profile | 深拷贝后原bank参数/buffers/flags/grad均保持；故意禁用拷贝必须检测到改变 |
| Numeric | FP32源端点先转FP64再差分；native endpoint=gain；无效mask/scale/NaN拒绝；Q相同 |
| Feature legality | 改truth、真实candidate output、oracle score，feature不变；在预测入口直接禁止这些路径访问 |
| Nested CV | 一个outer验证折的labels毒化后，该折fit+pred bitwise不变；其他折若合法用其作train不要求不变 |
| Preprocessors | validation-only异常feature/label不能改变train scaler/PCA/inner HPO；outer和inner净化后全support不相交 |
| Freeze | 未commit/哈希不匹配/动作缺issue，scorer拒绝；每方法每issue恰好一份OOF |
| Decision | oracle大而合法收益0、static解释收益、CI跨阈值、少blocks、72h不满足、missing evidence，各分支返回正确限定状态 |

测试不得只手写一个期望结果字典；要经过真实薄runner/reader/worker入口，spy断言拒绝发生在trainer/forward/数据读取之前。轻量替身可以替代昂贵模型，不能mock掉正在测试的guard。

## C. GPU
历史S0 1/4/12 PASS与Fs/bank认证不重跑，除非核心source/后端/资产确实改变并经新授权。
本轮C-J1：8已曝光issue×6轨迹（5候选+F0）×两worker复算；endpoint、历史可比cell和bank不变性；运行元数据完整。
C-J2：一次冻结DEV×5矩阵及F0背景，4分片；全部完备、同identity且可CPU重算才发布。

## D. 科学验证
GPU作业成功不能直接记weather utility成功。需完整OOF动作、相同分母native指标、Fs/static/F0配对差、oracle限定、独立时间blocks与成本。每个bootstrap重采样重算Fs归一化分母。条件于本次OOF策略的CI不包装成整个训练算法跨样本的保证。
最终确认未授权；双头/响应模型比较不在本轮默认任务。所有未来测试/实验结果标TO_BE_RUN，不能在日志里预填passed数字。
