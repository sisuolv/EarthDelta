# Artifact Contract

统一根：`artifacts/<phase>/<UTC-run-id>/`，目录不得覆盖。P0-01的只读审查输出可放外部工作目录。不要把weights、大数据、密钥、token化下载URL自动提交到GitHub。

每次最少：
```text
config.json
manifest.json
metrics.json
summary.md
stdout.log
decision.json
```
实际有数据则增加candidate_outcomes.parquet、predictions/index.json与对应Zarr/NPY；actualfigure放figures/并记录源表hash。没有运行只写status，不创造假prediction或sciencefigure。

manifest字段：run_id、role(dev/calib/confirm/oracle/serving)、git_head、dirty_diff_sha256、checkpoint_sha256、static_ref_sha256、bank_sha256、candidate_manifest_sha256、metric_spec_sha256、split_sha256、grid_sha256、normalization_sha256、command、seed、versions、device、dtype、started/finished UTC、exit_code、cost coverage。

metrics含数值与有效性，不允许NaN裸JSON：未定义项写null并解释reason/denominator。decision含状态、preconditions、effect/CI/δ、样本与功效、前置任务证据、next_allowed_tasks；不得只写passed=true。

每行候选结果：(issue_id,process_group_id,plan_id,lead,variable,reference_role,response_kind,predicted_gain,actual_gain,total_cost,feasible,complete,invalid_reason)。真值或oracle字段不可进入servinginput文件。

报告对失败保持三层：技术失败/资源缺失；开发负结果；独立确认负结果。更正评分时新增corrected派生表，保留原数据、原回答与旧表和原因。
