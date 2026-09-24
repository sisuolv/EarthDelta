# Artifact 合同

## 三种状态
- `OBSERVED`：从实际源码、计算记录、原始/嵌入式job_result或测试日志得到的已知事实。必须另列来源类型。
- `TO_BE_RUN`：未来预期结果，不得附拟造的数字。
- `MISSING`：必须的文件、字段或证据未取得；若阻断前驱，任务状态另记`BLOCKED`。

`PASS`、`FS_SELECTED`、`BANK_CERTIFIED`是各自协议的判决，不是证据等级。不能把只有job提交、准备好的argv或dry-run当作GPU完成。

## 每个节点和每个GPU尝试必存
```text
RUN/
  protocol/{fp05_protocol.json,fp05_protocol.sha256,split_freeze.json,
            sampling_declaration.json,dev_issue_ids.json,new_source_manifest.json,authorization.json}
  jobs/<job-name>-<attempt>/
    submission_record.json
    invocation.json
    source_manifest.json
    environment.json
    job_result.json
    stdout.log
    stderr.log
    payload_result.json
  cache/<issue-id>/
    issue_manifest.json
    legal_features.npy
    forecasts.npy
    F0_background.npy
    sealed_targets.npz
  policies/{folds.json,inner_folds.json,fit_records.json,oof_predictions.json,
            prediction_freeze.json,access_log.json}
  evaluation/{dev_results.csv,paired_block_bootstrap.json,same_panel_F0.json,
              independent_recompute.json,costs.json,failures.json,claims.json}
  decision.json
  HANDOFF.md
```

任务可按物理分片存储，但必须提供稳定的逻辑索引。不要为了凑文件名复制一个原始JSON到多个互不相符的任务中。

### job_result必需字段
`job_id`（可由同run_id的原始submission companion无歧义联结）、`run_id`、`hostname`、`started_utc`、`finished_utc`、`returncode`、`elapsed_seconds`、平台状态、payload命令/返回码、source snapshot/hash、协议hash、设备和软件版本。返回码成功与payload判决分列，不能用SUCCEEDED掩盖内部FAIL。

已读的X1 results JSON包含嵌入式job_result，可作为相应执行记录；S0/Fs-v2/bank的task_result字符串不能替代尚缺的原件。没有原件时保存MISSING，不从字符串生成假平台记录。

### 全链条身份
每个产物绑定：源码commit+实际import路径/hash、17个保护source pin、新增call graph源码、完整F0/Fs/bank文件SHA和state digest、S0 config与certificate SHA、Fs/bank资格与assembly证书、normalization原资产和有效身份、grid坐标值/变量序、native Q和scale、候选有序列表与真实系数、split ledger、实际UTC支持时次、内容证书、fold/seed/超参数、父产物hash。

SHA必须注明是Git blob SHA、文件SHA-256还是state/content digest；禁止混用。文件名或一个`status: PASS`不能认证身份。缺expected hash默认拒绝，不能自算后把它补成expected来“认证自己”。

### 候选与分母
每个issue必须有5个候选（Fs/no-edit+4 singleton），每个候选有6/24/72h完整字段；F0为旁路背景，不在candidate registry。每条记录有`planned/attempted/completed/failed/fallback`状态及reason。任何科学计分必须使用相同冻结issue分母，不做successful-subset评分。

### 预测冻结与标签封存
每个outer fold拥有只读训练标签列表和不含truth/候选实际响应的validation特征视图。所有模型选择、normalizer、PCA、regime、ridge调参在相应训练层。先保存模型、全OOF动作和prediction_freeze SHA，再由evaluator打开验证标签。封存文件名不是访问控制；必须有生产调用的role check和访问日志，测试验证不读取。

### 完整性和发布
先写staging、记录文件hash、验收完整矩阵，再原子发布`committed=true`；失败产物保留但标不可消费。每次retry新attempt目录，不覆盖旧失败。raw数值与是否通过分开，后续决策从数值重算。

### 数值边界
S0的1e-5不能移作其它一切检查的容差；A1-A5保持0。同GPU、相同路径debug复算应满足预登记exact锚点。CPU与GPU的不同reduction顺序不应被暗称bitwise一致：用事前算术测试冻结的FP64重算误差规则，记录绝对/相对误差，不事后迁就结果。
