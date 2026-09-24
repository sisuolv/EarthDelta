# 复现与身份

## 审查起点

repository `sisuolv/EarthDelta`，branch `audit/round2-review-20260921`。
读取HEAD `08093650ba56e8cf709d75a61c26c50a0c35f6f7`；受审实现/FP05计划 `e0136d4ca8ad3e49f49bdb4947cc73a95f082c4a`；上一实现a3596e6，旧基线d749a1c。
新HEAD仅文档不同，已比较earthdelta/scripts/tests/reference树。启动时仍重新记录真实HEAD和dirty diff，不reset/stash用户工作。

## 已认证运行条件

S0记录torch `2.3.1+cu121`，xformers `0.0.27`，H10080GB，official模型路径，FP32/TF32关闭。
input inverse使用原始NPZ dtype后构造inverse；源、资产和计算选项变化即新执行身份。
实际权重文件SHA与state digest分别保存，不能仅凭相同seed认定模型相同。
S0/Fs/Bank cert中的实际source SHA是消费依据，不用commit message替代工作区snapshot。

## 新代码策略

新增split_freeze/candidate_cache/policy_oof/runner等薄模块，不修改FP04的17个pin。
没有复制另一条shared-F0 protocol；保持certified fitted-Fs reference与continuation。
现有Fs-only admission消费者拒绝policy_dev是正确行为；新缓存须建立自己严格的policy_dev消费者，不能修改role绕过。

## 数据与曝光

为所有真实读取保存完整时间支持、indices、坐标、变量顺序、归一化、内容hash与role。
X1_N32/N64在2020H2的训练/输出用于设计分析；被舍弃模型也算研究曝光。
不同分辨率/变量/时间窗口不能换名复用certificate。
2019H2 dev只作回顾性开发，2020H2不是clean confirm；未指定确认窗口不得读confirm。

## 随机性与复现

记录真实Python/NumPy/Torch seeds、TF32 getters、matmul precision、模型eval、优化器/HPO参数、fold生成及bootstrap seeds。
四次bitwise相同训练不增加独立样本数。未来环境不同，不承诺永久bitwise；必须重新按声明范围验证。
CPU线性代数版本、线程数、BLAS也记录；双worker比较前同环境，不用allclose偷偷替代bank exact规则。

## 重试与成本

使用已配置的CCI提交器，保存原始argv、scheduler receipt、source manifest、stdout/stderr、返回码和耗时。
本包未提供也未调用CCI凭证；不要把命令字符串当提交记录。
失败不覆盖原目录；只允许预登记infra retry，保留旧ID/旧hash/偏差说明。
实测profile决定预算；不伪造estimated_minutes、显存或速度。固定N以后不得因CI调整规模。

## 本包验证

`tools/validate_package.py`只检查包结构、JSON、依赖、初始授权和SHA；没有资格签发天气PASS。
`tools/check_task_evidence.py`只读本地已有metadata/receipt并记录hash，不提交GPU、不运行训练。
