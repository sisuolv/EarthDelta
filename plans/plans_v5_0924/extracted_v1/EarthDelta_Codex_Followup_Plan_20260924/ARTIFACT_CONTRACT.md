# Artifact合同：口头PASS不构成完成

## 1. 每项task_result的公共字段

```json
{
  "schema_version": "ed-fp05-task-result/1",
  "task_id": "FP-05a",
  "status": "OBSERVED | TO_BE_RUN | BLOCKED | MISSING",
  "verdict": "PASS | FAIL | INCONCLUSIVE | BLOCKED",
  "started_utc": null,
  "finished_utc": null,
  "source_commit": null,
  "source_manifest_sha256": null,
  "protocol_sha256": null,
  "predecessors": [],
  "commands": [],
  "job_id": null,
  "job_result_path": null,
  "returncode": null,
  "elapsed_seconds": null,
  "evidence_outputs": [],
  "failures": [],
  "scope": "exact scope, not a paper claim"
}
```

空模板未来文件只能标TO_BE_RUN，不能写PASS；代码/测试/执行/科学四层分别存status。
CPU任务job_id可null并注明NOT_APPLICABLE，不造GPU编号。GPU任务必须有真实job_result，含run_id/主机/开始结束/rc/耗时，和调度ID→run_dir映射。
stdout/stderr、argv、source snapshot、Torch/CUDA/xformers/TF32 getters、设备、实际输入文件SHA、失败重试全部保存。

## 2. 前驱身份

S0 gate SHA = `5afbe78601afa5caa938ed4b8a606d3b98e633719f4b204f0fdd48594135d438`（记录值，需在执行机重哈希）。
Fs merged state digest = `370bd3bcc7545755bcdb60e482e1ce1ebc201302dc710f7992a4d76c4915bb97`。
Bank state digest = `13bc905444e4af1008b95d8820f0db4a5585621c712b1330d096b027df9cb33b`。
Bank file SHA = `c3e3e34b60b463b41a6c824fa680d844a2830cd2c7b40d3fc4544dc985b85f81`。
Registry file SHA = `c480684df1e9068387d347e00f6afa16f2b08db2d137d23175fb3933d944affd`。

file SHA、state digest、ArtifactVersion digest是不同对象，不可互相代换。本包未重哈希未取得的权重。
加载必须验证certificate、内部identity和实际参数；不能仅验证文件名、摘要字符串长度或PASS标签。
17个核心源pin以原bank protocol中的映射为准。新脚本/新增模块另建source manifest，所有cache/OOF消费者核对；只记不查不算闭环。

## 3. 曝光与split

每条记录包含issue UTC、真实history/target indices、完整support、data role、exposure kind、来源job/artifact、source status（OBSERVED/MISSING）、是否进入梯度/结果筛选。
X1必须关联实际训练record而不是只读manifest候选。已曝光不是通过删除权重可撤销的状态。
`confirm_window=null`与`confirm_access_authorized=false`都应令confirm consumer拒绝；不是允许任意日期。
新split manifest引用完整exposure ledger SHA。fold/manifests签名冻结后不得只换文本窗口。

## 4. Cache

每个N个issue有5个candidate、3个leads；F0单独报告。绑定：Fs/bank/registry/protocol/split/Q/source/env/input-content hashes。
每条保存 `issue_id, candidate_id, lead_hours, status, native_loss, forecast_path/hash, seconds`。
损失按统一空间/权重/scale；完整forecast端点保存或使用预登记可复核格式，不能只输出一张均值表。
`legal_features`文件与label/real candidate forecast拆分；selector loader不得打开禁止文件。文件名sealed不是权限证明。
merge核对完整集合、重复、finite、shape、每个hash，不能把缺失行和模型失败自动删除。所有尝试进入failure ledger/costs。

## 5. OOF freeze和评分

每fold保存训练/验证support列表及净化理由；inner folds、每次scaler/PCA/kmeans/ridge拟合的issue集合；HPO所有尝试；seed和择优规则。
静态选择必须每个outer train独立做。无样本/平局回退no-edit规则预先固定。
freeze含预测内容SHA、每issue唯一OOF candidate ID/所有预测gain、head/transform哈希、fold/source/protocol/registry等；写入后只读。
评分器在打开结果标签前先验证freeze完整与只读；无freeze、不完整或有泄漏则失败。
F0背景不进入任何selector训练或候选域。

## 6. 统计与报告

报告每个方法的三时效loss、Fs-relative gain、OOF-static增量、same-issue F0净收益、72h损害、全部失败分母和成本。
bootstrap保存block ID/界限、seed、replicate index或可重建随机状态、实际有效block数、CI、阈值/分母。
固定预测的bootstrap标conditional，CPU refit/bootstrap敏感性单列；一次OOF不等于128个独立试验。
成本计入背景轨迹、debug、失败、重试、头拟合/搜索，不以任务运行退出0代替结果有效。
