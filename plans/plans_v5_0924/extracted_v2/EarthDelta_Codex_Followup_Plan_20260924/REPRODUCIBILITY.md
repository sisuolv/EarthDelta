# 复现与证据保全

## 冻结身份
- 受审源码/计划提交：`e0136d4ca8ad3e49f49bdb4947cc73a95f082c4a`；观察分支HEAD：`08093650ba56e8cf709d75a61c26c50a0c35f6f7`，该后一提交仅文档。
- 历史实现包：`bc57a70`；之前实现：`a3596e6b804e9d23b72d1247b08c47129d6b50b1`。
- 第一个动作是`git status`、`rev-parse HEAD`及diff，只读，不reset、不checkout覆盖用户改动。
- 按当前认证`bank_protocol_v1.json`的source_at_preregistration重hash17个核心文件。保留同一实际import backend，不只记录版本号。新split/cache/OOF文件与依赖另建manifest并冻结。

## 环境
历史认证torch2.3.1+cu121、xformers0.0.27、H100。记录TF32有效getters和float32_matmul_precision；不能仅看环境变量就推断运行时状态。不要盲目升级共享环境，不伪造Lightning或SDPA冒充official。
同一case的四个复制结果相同，只说明这些实际运行可重复，不保证所有backends/硬件/优化器确定性。

## 数据与曝光
曝光记录必须覆盖所有被试过的模型，包括X1和未选checkpoint。只做完整性检查的读取与用于拟合/模型选择/观察效果的读取分列。全部读取记录UTC起报与完整支持、grid/channel、内容SHA、role、reader、目的。
2019H2 DEV是回顾性实验；2020H2已经被X1使用，不得称项目级untouchedconfirm。新confirm位置本轮不指定、不读取。
数据筛查的接受总体是通过预先完整性标准的实际样本，不声称代表被排除的缺失/异常天气。

## 预登记
先冻结选点和替换规则，再扫描完整性；代码/方法/阈值/预算冻结先于forecast outcomes，最终N和exact list在debug实测成本之后、DEV forecast之前只冻结一次。
协议SHA必须由独立的授权/提交记录消费，不能写到自身形成循环。保留时间戳、原提交、source bundle、旧版本和偏差文件。内部文件时间戳/sha支持时间顺序，不是外部可信时间戳证明；不回填或篡改。

## 执行
run_id唯一，attempt单独目录。保存submit argv与真实job_result、payload result和原始stderr。任何失败不覆写；infra与科学失败分开；临时hash、空目录或dry-run不等于执行。

## 统计
固定模型/采样/CV/bootstrap seed并保存；各candidate同issue同fold；outer/inner完整support净化；数据依赖预处理仅train。固定N不自动证明power，报告有效blocks、预估MDE及假设。
主比较与辅助比较事前区分。不能从本轮DEV挑最佳方法后把同一DEV称为confirmation。下一次确认需要另行未曝光数据与冻结pipeline。

## 再认证触发
若保护的core17、F0/Fs/bank权重、normalization源、变量序、grid或数值后端发生改变，停止并给变更影响说明；只重开受影响的门。缺平台回执但数值产物存在时优先补文件，不自动重跑模型。

2019也是所用Stormer checkpoint-selection的历史验证年；本轮DEV的“未见”只能相对于本轮Fs/bank/policy选择而言，不是整个预训练/模型选择流程的最终未触碰测试集。
