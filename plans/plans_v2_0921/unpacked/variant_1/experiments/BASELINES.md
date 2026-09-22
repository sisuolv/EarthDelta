# BASELINES — 一个迭代的最小对照

所有方法共用合法I、目标Q、候选资格、reference及split；oracle信息不送入deployable forward。输入/仿真权限与实际成本是两件事，均必须记账。

| ID | 方法 | 当前作用 / 任务 | 输入、拟合和输出 | 复用与公平约束 |
|---|---|---|---|---|
| REF | no-edit=F_ref=Fs | R2I-02/03必需 | 固定参考预报；gain=0 | F0仅原始背景，不能把弱F0冒充强静态参考；hold后也继续Fs。 |
| STATIC | dev-selected best-static | R2I-03必需 | 只在dev_select选一个candidate，此后固定 | 记录训练充分性；不可在dev_probe/confirm重新挑最优静态。 |
| REGIME | 简单regime router | R2I-03必需 | 用合法低维/固定元信息划分；各组选择仅fit/dev_select | 不新增Complexity Atlas；与其他方法相同candidate/budget；分组样本不足应预设fallback。 |
| DG-LITE | ridge/小MLP direct-gain | R2I-03必需 | I和candidate descriptor→scalar gain；按组out-of-sample | 尽量复用heads/低维特征，HPO选择不能触及probe标签；不读取exact du。 |
| ORACLE | complete finite hindsight oracle | R2I-03必需、仅离线 | 全部注册实际候选loss→逐起报最佳 | 不声称deployable；完整表、共同成本可行集；不以top3/中心差分替代。 |
| PAIR | e0/du + analytic gain | R2I-04条件 | 现有ComposedPredictionHead简化配置 | JEPA/memory/calibration/variance额外项默认关闭或明确消融；同Q_eff。 |
| PROJ | 决策投影/条件矩 | R2I-04条件 | fit固定响应基上的误差，或有限候选相关条件矩 | 投影只fit；oracle span/b/H不得作为新起报输入；不声称通用新算法。 |
| DG-FAIR | direct-gain +同等响应辅助 | R2I-04条件 | 同合法I、描述符、可用无标签仿真；辅助训练+gain | 一份核验truth可标所有candidate；同仿真额度、核验origins及HPO总账。 |
| EDITED | direct edited-forecast（residual/skip） | R2I-04仅cap允许 | 对candidate直接预测未来端点，使用合理公共reference skip | 不故意用更重裸预测网络作为弱对照；若不跑记NOT_RUN，缩窄比较结论。 |
| FOUR-CELL | 真/预测e与du四格 | R2I-04仅诊断 | 在评分侧替换目标量，同actual outcome结算 | true du包含运行所有candidate成本；oracle e看未来，不是serving。 |
| VIRTUAL | reference+predicteddu | R2I-05条件 | 对相同选中action生成全场响应，添加参考 | 摘要响应需公平full-field readout；缺此条件记NOT_COMPARABLE_SUMMARY_ONLY。 |
| FEEDBACK | 逐步多变量输出纠正 | R2I-05条件 | Fs(x)+C(I,x,step)，修正状态反馈后续 | 与参数编辑同hold/continuation/信息和多步训练资格；不能只做末端单步纠正。 |
| RESIDUAL | reference+predictederror | R2I-05可共享读出 | 同未来lead的全场残差纠正 | 在相同输出与成本条件下成立；非本轮必须额外训练大网络。 |

## 推荐实现复用

- `ExpertLoRA`、`WeatherStepBridge`、`controlled_rollout`：已有天气执行链，最小补丁。
- `cached_responses`：借用同初态有限候选执行语义，扩成真实trajectory cache，不改为局部Jacobian教师。
- `ComposedPredictionHead`：已有e/du及lead接口，先修明确配置/Q，不重写新JEPA架构。
- `numpy/scipy/torch`：已有环境下的小回归/优化，不新增商业求解器或大型训练框架。
- 已有上游Stormer和评测代码只用已核验的本地版本；缺依赖记录BLOCKED，不自动联网安装。正式简单Q评分不以先装完整WeatherBench-X为硬门。

## 必須记录

每种方法的信息字段、仿真origin与candidate数、核验origin数（去重）、bank共享成本、额外训练与HPO、preview/重算/控制器/执行成本、失败与fallback。分别比较同标签与同总成本前沿；不能声称这些轴自动全部相等。

## 不能据此宣称

DG-LITE失败不否定全部合法policy；oracle高不说明可部署；head MSE更低不等于选择更好；输出纠正无限表达等价不决定有限预算胜负；候选更多带来oracle提高不证明novelty。
