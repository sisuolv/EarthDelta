# EarthDelta

Research plan and review materials for **EarthDelta**: response-space planning of low-rank edits for a frozen ML weather model (Stormer, 1.40625°). Status (2026-09-19): planning and literature stage; no weather experiments have been run yet.

## Layout

| path | content |
|---|---|
| `plans/plans_v1_0919/EarthDelta_v5_Research_Kit.zip`, `EarthDelta_ResponseKit_20260917.zip` | the two original v1 plan kits (CPU-tested primitives + plan documents) |
| `plans/plans_v1_0919/extracted/` | the same kits, extracted (41 + 50 CPU unit tests) |
| `plans/plans_v1_0919/EarthDelta_v6_Review_and_Plan_CN.md` | review of both kits, novelty analysis against 5 literature surveys, merged v6 plan (Chinese) |
| `plans/plans_v1_0919/literature/survey_A–E.md` | raw literature / tooling survey reports with arXiv IDs and dates (English) |
| `plans/plans_v1_0919/v6_draft/` | `research_spec_v6.yaml`, `PACKAGE_MERGE_MAP.md`, `cds_request_template.py` |
| `plans/plans_v1_0919/chatgpt_review_packet/` | prompt + single-file packet for an external LLM red-team review |
| `reference/` | `README.md`, `_manifest.json` (pinned HEADs + licenses) and `_clone_refs*.sh` for 46 third-party reference repos; the clones themselves are git-ignored |

## Running the kit tests

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python   # needed in the NVIDIA torch container
cd plans/plans_v1_0919/extracted/v5_Research_Kit/EarthDelta_v5_kit && PYTHONPATH=. python -m pytest -q tests
cd ../../ResponseKit/EarthDelta_ResponseKit_20260917 && python -m pytest -q
```

## One-sentence claim (current)

Amortize forecast sensitivity to parameter edits into an issue-time controller: learn the label-free response `du` of a frozen forecaster to each low-rank edit and a flow-dependent error forecast `e0`, select edits under a budget with the exact FSO-form quadratic gain `2<e0,du> - ||du||^2`, and show transfer to new budgets, dictionary members and lead times that end-to-end conditional adaptation cannot provide. See the v6 plan for what is and is not claimed as novel.
