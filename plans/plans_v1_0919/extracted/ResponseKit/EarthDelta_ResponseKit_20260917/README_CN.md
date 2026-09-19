# EarthDelta Response Kit：响应与修正交互原型

日期：2026-09-17。**本包是 CPU 验证过的研究组件，不是完成训练的天气模型。**

## 1. 已完成与未完成

本轮实际实现并测试：纯函数响应探测、响应几何、有界离线教师、非线性候选复验、显式程序支持、直接系数学生头、修正收益/交互学生头、无未来标签的低维在线代理规划器。测试结果见 `TEST_REPORT.txt`，环境见 `EXECUTION_STATUS.json`。

没有完成：真实 Stormer 权重加载与注入、ERA5 下载和天气训练、历史数据驱动器、已训练的学生、GPU 性能测试、真正跳过适配矩阵乘法、WeatherBench-X 接入、跨模型迁移。不要把合成例子写成天气成绩。

原 `EarthDelta_Starter_20260915.zip` 的21项测试另行重新运行通过。新包的测试不是原版与 Stormer 的集成测试，两套通过数不能证明真实天气集成正确。本包不覆盖旧文件，不含任何上游仓库源码或权重副本。

## 2. 文件与依赖

| 文件 | 实际提供的接口 |
|---|---|
| `earthdelta_response/probe.py` | `central_response`, `local_linearity_error`；单样本总d维程序的中心差分 |
| `geometry.py` | `ResponseGeometry.from_error`, `response_distillation`；H=RᵀQR，b=RᵀQe |
| `teacher.py` | `box_candidates`, `verify_candidates`；**仅训练/离线侧允许标签** |
| `program.py` | `Slot`, `ProgramSpec`；从总程序系数展开到明确时间/层/区域支持 |
| `student.py` | `BoundedProgramHead`；历史特征到有符号有界系数 |
| `utility.py` | `InteractionUtilityHead`, `plan_from_prediction`；历史特征预测b/H，枚举小支持并解盒约束QP |
| `examples/synthetic_probe.py` | 小型代数例子；不是ERA5或PDE求解器验证 |
| `configs/research_spec.yaml` | 待接入天气代码的研究规格；**不是Stormer CLI可直接运行配置** |
| `CODEX_PLAN_CN.md` | 研究设计、真实代码接入点、任务顺序与验收 |

直接运行测试需要兼容的 Python、PyTorch、NumPy、SciPy、pytest。不要为测试覆盖正在使用的 CUDA/PyTorch/xFormers 环境。`pyproject.toml` 的最低版本不是全部版本的兼容性声明，只有执行状态记录中的组合在本轮测试。

在包根目录运行：

```bash
python -m pytest -q
PYTHONPATH=. python examples/synthetic_probe.py
```

可在隔离环境使用 `python -m pip install -e .`。已有兼容研究环境可以使用 `python -m pip install --no-deps -e .`，避免自动替换 CUDA 相关依赖；缺失依赖应在隔离环境补齐。

## 3. 两种流程不能混用

训练/离线：

```text
冻结backbone + 已训练的非零bank + 合法历史
  -> central_response(callback, reference)
  -> 用训练未来标签构造 ResponseGeometry
  -> box_candidates
  -> 对少数候选完整非线性复验
  -> 得到教师样本、失败回执与无修正结果
```

推理：

```text
合法历史 -> 已训练的状态编码器
        -> BoundedProgramHead -> 有界程序
或者    -> InteractionUtilityHead -> 预测b/H
        -> plan_from_prediction -> 有界程序
        -> 待实现的Stormer显式受控rollout
```

`plan_from_prediction` 不需要真实未来、离线响应R或新的天气探测。但它只是一个小型数值规划器；是否有预报收益取决于尚未训练验证的预测b/H。它的无修正候选不提供真实误差不增加的保证。

## 4. 重要数值与科学边界

- d是完整程序的总自由度，不是每patch的自由度。d=8的中心差分是17条轨迹；默认重放一致性检查另加1条，共18。多步轨迹不是单步模型调用。
- `central_response` 会检查输出有限性、系数输入未被修改、基线重复一致性；无法替调用者证明不存在隐藏状态。请用不可变初始快照、eval模式和可复现随机设置。
- double差分累积不能恢复低精度模型前向已经丢失的信号。必须检查epsilon变化、混合方向和非线性实际收益。
- `box_candidates` 是分量盒约束，不是L2球约束；`bound`限制offset，非零参考a0时还必须检查总系数a0+offset可行。本首版规格使用a0=0。
- 训练教师可用完整真值进行候选拒绝；推理侧不可复制这种含未来标签的拒绝步骤。
- H是局部响应Gram矩阵，非对角项表示响应重叠；它不等于完整非线性损失Hessian，也不能证明大气因果关系。
- 字典B全零时对系数的响应全零。先训练非零bank，再建立响应。
- `ProgramSpec.coefficients_for`只展开dense系数，**没有跳过矩阵乘法**。当前cost是声明的加性单位成本，不是实测GPU成本。
- 初版学生是起报时一次产生整段程序。在线按新预测前缀重规划、执行账本和掩膜仍需在天气集成中实现。
- 预测H的PSD参数化是一种结构约束，不保证它接近真实响应，更不保证选出的程序有正收益。

## 5. 最先交给Codex的任务

先读 `CODEX_PLAN_CN.md` 的任务R00—R03。保留旧starter的原始21项测试，将旧StateGatedLinear的显式coefficients接口接到Stormer的指定attn.proj。完成真实checkpoint零增量等价和非零bank响应探针后，再做教师/学生实验。

不要把本包作为“已经有训练脚本”的完整项目；新增文件和待实现文件在计划里分别标明。
