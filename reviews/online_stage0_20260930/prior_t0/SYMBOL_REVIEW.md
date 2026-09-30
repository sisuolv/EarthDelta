# T0 符号核验

基于实际源码的 AST；函数体不执行。局部函数存在不等于可作为模块接口导入。

| 路径 | 符号 | 行 | 函数签名 / 类型 | 局部函数 |
|---|---|---:|---|---|
| `earthdelta/probe/rollout.py` | `LoadedBridge` | 27 | `ClassDef` | False |
| `earthdelta/probe/rollout.py` | `load_bridge` | 33 | `checkpoint: str &#124; Path, norm_dir: str &#124; Path, device: torch.device` | False |
| `earthdelta/probe/rollout.py` | `rollout_trajectory` | 49 | `bridge: WeatherStepBridge, x_norm: torch.Tensor, *, steps: int, deltas: Mapping[int, torch.Tensor] &#124; None=None, target_blocks: Sequence[int]=TARGET_BLOCKS, differentiable: bool=False, checkpoint_steps: bool=False` | False |
| `earthdelta/probe/rollout.py` | `area_weighted_mse` | 87 | `pred_raw: torch.Tensor, truth_raw: torch.Tensor, latitude: np.ndarray, channel_index: int` | False |
| `earthdelta/probe/rollout.py` | `l6_per_issue` | 98 | `pred_norm: torch.Tensor, truth_raw: torch.Tensor, bridge: WeatherStepBridge, latitude: np.ndarray, variables: Sequence[Mapping[str, object]], leads: Sequence[int], denominator: Mapping[str, float]` | False |
| `earthdelta/probe/rollout.py` | `l6_cells` | 121 | `pred_norm: torch.Tensor, truth_raw: torch.Tensor, bridge: WeatherStepBridge, latitude: np.ndarray, variables: Sequence[Mapping[str, object]], leads: Sequence[int], denominator: Mapping[str, float]` | False |
| `earthdelta/probe/edits.py` | `DirectWeightEditor` | 22 | `ClassDef` | False |
| `earthdelta/probe/edits.py` | `DirectWeightEditor.__init__` | 25 | `self, model: torch.nn.Module, deltas: Mapping[int, Tensor], *, blocks: Iterable[int]` | False |
| `earthdelta/probe/edits.py` | `DirectWeightEditor.__enter__` | 33 | `self` | False |
| `earthdelta/probe/edits.py` | `DirectWeightEditor.__exit__` | 48 | `self, exc_type, exc, tb` | False |
| `earthdelta/probe/edits.py` | `pristine_state_digest` | 12 | `model: torch.nn.Module` | False |
| `earthdelta/probe/contracts.py` | `issue_index` | 101 | `issue_time: str, *, year: int=2020` | False |
| `earthdelta/probe/access.py` | `ProbeAccessController` | 24 | `ClassDef` | False |
| `earthdelta/probe/access.py` | `ProbeAccessController.read` | 112 | `self, path: str &#124; Path, indices: Iterable[int], *, purpose: str, stage: str, job: str='local'` | False |
| `earthdelta/probe/access.py` | `ProbeAccessController.authorize` | 64 | `self, path: str &#124; Path, indices: Iterable[int], *, purpose: str, stage: str, job: str='local'` | False |
| `earthdelta/probe/estimators.py` | `per_issue_oracle` | 18 | `values: np.ndarray` | False |
| `earthdelta/probe/estimators.py` | `static_best_arm` | 30 | `values: np.ndarray` | False |
| `earthdelta/probe/budget.py` | `BudgetLedger` | 16 | `ClassDef` | False |
| `earthdelta/probe/budget.py` | `BudgetLedger.ensure` | 41 | `self, gpu: str, projected_card_hours: float` | False |
| `earthdelta/probe/budget.py` | `BudgetLedger.record` | 48 | `self, *, job_id: str, gpu: str, card_hours: float, stage: str, status: str, projected: bool=False, **extra: Any` | False |
| `earthdelta/bridge/stormer_bridge.py` | `DEFAULT_VARIABLES` | 40 | `Assign` | False |
| `earthdelta/bridge/stormer_bridge.py` | `controlled_rollout` | 1115 | `bridge: WeatherStepBridge, x_norm: torch.Tensor, variables: List[str], interval: int, steps: int, plan: EditPlan, expert_loras: Dict[int, ExpertLoRA], target_blocks: Tuple[int, ...]=(18, 19, 20, 21, 22, 23), sparse: bool=False, return_trajectory: bool=False, differentiable: bool=False` | False |
| `earthdelta/bridge/stormer_bridge.py` | `_assert_no_activation_checkpointing` | 997 | `model: nn.Module` | False |
| `earthdelta/static_adapter.py` | `build_fs_adapter` | 477 | `hidden_size: int, target_blocks: Sequence[int]=DEFAULT_TARGET_BLOCKS, *, rank_per_expert: int=4, scale: float=1.0, seed: Optional[int]=None, init_up_std: float=0.0` | False |
| `earthdelta/static_adapter.py` | `fs_edit_plan` | 537 | `steps: int, *, plan_id: str='fs_always_on'` | False |
| `earthdelta/static_adapter.py` | `area_weight_q` | 208 | `lat: Sequence[float] &#124; Tensor, dtype: torch.dtype=torch.float64` | False |
| `earthdelta/lowrank.py` | `ExpertLoRA` | 17 | `ClassDef` | False |
| `earthdelta/lowrank.py` | `ExpertLoRA.__init__` | 29 | `self, in_features: int, out_features: int, num_experts: int=8, rank_per_expert: int=4, scale: float=1.0, shared_down: bool=False, shared_up: bool=False` | False |
| `earthdelta/lowrank.py` | `ExpertLoRA.forward` | 157 | `self, x: Tensor, coefficients: Tensor, sparse: bool=False` | False |
| `earthdelta/memory.py` | `VerifiedRecord` | 18 | `ClassDef` | False |
| `earthdelta/memory.py` | `VerifiedRecord.__post_init__` | 32 | `self` | False |
| `earthdelta/memory.py` | `eligible_records` | 47 | `records: list[VerifiedRecord], origin: int, version: str, blocked_events: frozenset[str]=frozenset()` | False |
| `earthdelta/memory.py` | `ewma_error_feature` | 128 | `records: list[VerifiedRecord], origin: int, version: str, decay: float=0.95, max_records: int &#124; None=None, blocked_events: frozenset[str]=frozenset()` | False |
| `earthdelta/v8/standard_metrics.py` | `_weights` | 64 | `latitude: np.ndarray, shape: tuple[int, int]` | False |
| `earthdelta/v8/standard_metrics.py` | `issue_mse` | 75 | `forecast: np.ndarray, truth: np.ndarray, latitude: np.ndarray` | False |
| `earthdelta/v8/standard_metrics.py` | `relative_rmse` | 90 | `forecast: np.ndarray, truth: np.ndarray, baseline: np.ndarray, latitude: np.ndarray` | False |
| `earthdelta/v8/standard_metrics.py` | `paired_bootstrap` | 108 | `forecast: np.ndarray, baseline: np.ndarray, truth: np.ndarray, latitude: np.ndarray, issue_times: Sequence[str], *, draws: int=10000, block_days: int=7, seed: int=0` | False |
| `earthdelta/v8/data_repair.py` | `repair_zarr_year` | 95 | `source_path: str &#124; Path, target_path: str &#124; Path, *, role: str, year: int, indices: Iterable[int], std: np.ndarray &#124; None=None` | False |
| `earthdelta/v8/data_repair.py` | `refetch_zarr_indices` | 164 | `target_path: str &#124; Path, *, year: int, indices: Iterable[int], max_batch: int=8` | False |
| `earthdelta/v8/data_repair.py` | `FailClosedRepair.record_access` | 39 | `self, *, role: str, year: int, path: str, operation: str, array_payload: bool` | False |
| `earthdelta/data/pull_wb2.py` | `ConservativeRegridder` | 318 | `ClassDef` | False |
| `earthdelta/wbx/evaluate.py` | `evaluate` | 196 | `prediction, truth, *, init_times: Sequence[Any], lead_times: Sequence[Any], metrics: Optional[Mapping[str, Any]]=None, aggregator=None, variables: Optional[Sequence[str]]=None, init_chunk_size: int=1, upcast_float64: bool=False, confirm_freeze_path: Optional[Union[str, Path]]=None, confirm_freeze_sha256: Optional[str]=None, authorized: frozenset=DEFAULT_AUTHORIZED_YEARS` | False |
| `earthdelta/wbx/evaluate.py` | `evaluate_single_chunk` | 269 | `prediction_chunk, truth_chunk, *, metrics=None, aggregator=None, init_times: Optional[Sequence[Any]]=None, valid_times: Optional[Sequence[Any]]=None, confirm_freeze_path: Optional[Union[str, Path]]=None, confirm_freeze_sha256: Optional[str]=None, authorized: frozenset=DEFAULT_AUTHORIZED_YEARS` | False |
| `earthdelta/wbx/evaluate.py` | `standard_metrics` | 132 | `climatology=None, *, wind_vector: bool=True` | False |
| `earthdelta/wbx/evaluate.py` | `make_aggregator` | 147 | `*, reduce_init_time: bool=True, regions: Optional[Mapping[str, Any]]=None, land_sea_mask=None, masked: bool=False` | False |
| `earthdelta/wbx/evaluate.py` | `assert_years_authorized` | 39 | `years: Iterable[int], *, confirm_freeze_path: Optional[Union[str, Path]]=None, confirm_freeze_sha256: Optional[str]=None, authorized: frozenset=DEFAULT_AUTHORIZED_YEARS` | False |
| `earthdelta/wbx/_vendor.py` | `ensure_vendored` | 333 | `*, check_clean: bool=True, need_jax: bool=True` | False |
| `earthdelta/wbx/_vendor.py` | `import_official` | 381 | `module_name: str, *, check_clean: bool=True` | False |
| `earthdelta/wbx/_vendor.py` | `add_wbx_deps_path` | 284 | `FunctionDef` | False |
| `earthdelta/wbx/export.py` | `unflatten_channels` | 132 | `data: np.ndarray, *, lat: np.ndarray, lon: np.ndarray, leading: Sequence[Tuple[str, np.ndarray]]=(), channel_names: Sequence[str]=tuple(CANONICAL_VARIABLES), attrs: Optional[Mapping[str, Any]]=None` | False |
| `earthdelta/wbx/baselines.py` | `PUBLIC_STORES` | 23 | `AnnAssign` | False |
| `earthdelta/heads.py` | `ReferenceErrorHead` | 106 | `ClassDef` | False |
| `earthdelta/heads.py` | `ReferenceErrorHead.__init__` | 112 | `self, feature_dim: int, target_dim: int, latent_dim: int=32` | False |
| `earthdelta/heads.py` | `ReferenceErrorHead.forward` | 126 | `self, context: Tensor` | False |
| `scripts/probe_p3_directions.py` | `target_params` | 71 | `model: torch.nn.Module` | False |
| `scripts/probe_p3_directions.py` | `flat_delta` | 78 | `vector: np.ndarray, params, device: torch.device` | False |
| `scripts/probe_p3_directions.py` | `load_issue` | 63 | `qc: dict, issue: str, *, cache_dir: Path, ledger: Path, run_id: str, job: str, stage: str` | False |
| `scripts/probe_p3_directions.py` | `calibration_effect` | 111 | `edited_norm: torch.Tensor, f0_norm: torch.Tensor, truth_raw: torch.Tensor, bridge, lat: np.ndarray` | False |
| `scripts/probe_p3_directions.py` | `main.effect` | 197 | `direction, a` | True |
| `scripts/probe_p3_directions.py` | `main.calibrate` | 204 | `direction` | True |
| `scripts/probe_p4_menu.py` | `main` | 34 | `FunctionDef` | False |
| `scripts/probe_step2_static.py` | `make_adapter` | 105 | `loaded, seed: int` | False |
| `scripts/probe_step2_static.py` | `state_cpu` | 116 | `adapters` | False |
| `scripts/probe_step2_static.py` | `restore_state` | 121 | `adapters, state` | False |
| `scripts/probe_step2_static.py` | `main` | 126 | `FunctionDef` | False |
| `scripts/probe_p1_qc.py` | `main` | 36 | `FunctionDef` | False |
