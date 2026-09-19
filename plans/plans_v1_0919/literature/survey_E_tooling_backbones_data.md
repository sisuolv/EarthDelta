# Survey E — engineering landscape: backbones, tooling, data access (verified 2026-09-19)

Source: background tooling agent, 2026-09-19; verified against local clones under `/mnt/afs/260010168/EarthDelta/reference/`, hf-mirror API, PyPI, and live GET probes. Items marked UNVERIFIED were not confirmed.

## 1. Stormer (tung-nd/stormer, arXiv 2312.03876)
- Repo frozen: `main` HEAD `58dfee5a6037399a40fefd492bc00421e0c885a8` (2025-03-17, "update readme"); no tags, no newer commits; 9 forks, none removes the xformers dependency (`stormer/models/hub/stormer.py:4`).
- Checkpoints on HF `tungnd/stormer` (HF sha `2ceebb74…`, 2024-10-23, license MIT): `stormer_1.40625_patch_size_2.ckpt` 5,625,590,679 B; `stormer_1.40625_patch_size_4.ckpt` 5,570,407,547 B. Lightning `.ckpt` (loaded via `checkpoint["state_dict"]`), size includes optimizer state (~400 M params model). `inference.py` defaults to patch_size_2. Mirror: `https://hf-mirror.com/tungnd/stormer/resolve/main/<file>`.
- 69 channels = 4 surface (t2m, u10, v10, mslp) + 5 vars × 13 levels; grid 128×256 = pole-free cell centres (`regrid_wb2.py:65-68`: lat −89.296875…+89.296875, lon 0…358.59375).
- Normalization: 10 npz files in `normalization_constants/` (109 keys each, 1979–2018). `inference.py` expects copies inside the preprocessed h5 dir.
- Architecture: DiT-style; per-variable `PatchEmbed` + variable-aggregation `nn.MultiheadAttention`; 24 blocks with `MemEffAttention` (`qkv` d→3d, `proj` d→d), timm `Mlp` (fc1/fc2), `adaLN_modulation` (SiLU + Linear d→6d); `FinalLayer` (Linear + adaLN 2d). hidden 1024, depth 24, heads 16. All adapter targets are plain `nn.Linear`.
- env.yml: python 3.11.5, torch 2.1.0 cu118, xformers 0.0.22.post7+cu118, timm 0.9.2, lightning 2.2.1. SDPA swap needs layout change (xformers (B,N,H,D) vs SDPA (B,H,N,D)).
- No Stormer-preprocessed ERA5 on HF (`tungnd/*` has only models, `stormer_forecasts`, IndiaWeatherBench).

## 2. Aurora (microsoft/aurora)
- Local HEAD `4765abc` (2026-09-14). HF `microsoft/aurora` (MIT code + weights; commercial use: contact Microsoft): 9 checkpoints incl. `aurora-0.25-pretrained.ckpt` 4.68 GiB, `aurora-0.25-small-pretrained.ckpt` 0.42 GiB, `aurora-0.25-v1.5.ckpt`, `-v1.5-ensemble`, `aurora-0.1-finetuned`, `-air-pollution`, `-wave`, `-12h-pretrained`; loader `Aurora.load_checkpoint(repo, name, revision, strict=True)`.
- GPU: ~40 GB for 0.25° inference (Aurora 1.5 ~32 GB); fine-tuning tested on 80 GB A100.
- LoRA built in: `aurora/model/lora.py` (`LoRA`: A kaiming, B zero, scaling alpha/r; `LoRARollout(max_steps=40, mode∈{single, from_second, all})`), applied only to Swin3D attention `qkv` and `proj` (`swin3d.py:130-139, 159, 182`); step index carried on `Batch.metadata.rollout_step` (`batch.py:43`, threaded at `aurora.py:435`). Defaults r=8, alpha=8, dropout 0, steps 40, mode single. MLPs, Perceiver encoder/decoder, patch embeddings NOT adapted. Base `Aurora` use_lora=True; `AuroraPretrained`/`Small`/`12h`/`V1p5` use_lora=False; `AirPollution`/`Wave` mode from_second.
- No coarse-resolution Aurora checkpoint exists → not a practical second backbone on limited GPU.

## 3. Other open-weight coarse backbones (PyTorch-friendliness)
| Model | Repo / weights | Grid | Licenses | Notes |
|---|---|---|---|---|
| ClimaX | microsoft/ClimaX `6d5d354` (2023-09-30); HF `tungnd/climax` `1.40625deg.ckpt` 443 MB, `5.625deg.ckpt` 432 MB | 128×256 and 32×64 | MIT / MIT | same grid as Stormer; timm ViT; patch 4 for [128,256] |
| ArchesWeather / Gen | INRIA/geoarches `b2bdb07` (2026-09-18); HF `gcouairon/ArchesWeather` `archesweather-m-seed0_checkpoint.ckpt` 340 MB, `archesweathergen_checkpoint.ckpt` 1.9 GB | 1.5°, 13×121×240 | BSD-3 / BSD-3 | `nn.Linear`-dense (qkv, proj, fc1/fc2, adaLN); on PyPI; **preferred second backbone** |
| ACE2 | ai2cm/ace `b648495`; HF `allenai/ACE2-ERA5` 1.8 GB | ~1° Gaussian 180×360 | Apache-2.0 / Apache-2.0 | SFNO MLPs are Conv2d(1×1), adapter needs Conv variant |
| Prithvi-WxC | NASA-IMPACT/Prithvi-WxC; HF 28.4 GB | MERRA-2 0.5°×0.625°, 2.3 B params | MIT / CDLA-Permissive-2.0 | too large |
| SFNO/FCNv2, FourCastNet 3 | NVIDIA/makani, earth2studio; NGC / HF `nvidia/fourcastnet3` 2.8 GB | 0.25° | Apache-2.0 | needs torch-harmonics; FCN3 needs 80 GB |
| Pangu-Weather | 198808xc/Pangu-Weather; ONNX ~1.1 GB | 0.25° | none / CC BY-NC-SA 4.0 | ONNX only |
| FuXi | tpys/FuXi; Zenodo 8.6 GB ONNX; HF `tpys/fuxi-2.1` PT2 3.9 GB | 0.25° | none / CC-BY-4.0 | frozen graphs |
| GraphCast small | google-deepmind/graphcast → redirects to google-deepmind/weathernext `f2f2c51`; `gs://dm_graphcast/params/GraphCast_small…npz` 144 MB | 1°, 13 levels | Apache-2.0 / **CC BY 4.0 since 2026-08-06** | JAX/Haiku |
| NeuralGCM | neuralgcm/neuralgcm; `gs://neuralgcm/models/v1/deterministic_{2_8,1_4,0_7}_deg.pkl` | 2.8/1.4/0.7° | Apache-2.0 / CC BY-SA 4.0 | JAX |
| AIFS-single | ecmwf/anemoi-core; HF `ecmwf/aifs-single-1.0` 0.99 GB | n320 (~0.25°) data grid | Apache-2.0 / CC BY 4.0 | PyTorch, heavy |
| WeatherMesh | windborne/weathermesh-3 | — | Apache-2.0 | no weights released |
Recommendation: ArchesWeather-M (1.5°, BSD-3 both) first, ClimaX 1.40625° (MIT, identical grid) second; ACE2 runner-up.

## 4. WeatherBench-X and WB2 data
- weatherbenchX HEAD `964a35e` (2026-09-17), Apache-2.0; **git-install only** (`pip install git+https://github.com/google-research/weatherbenchX.git`); deps include `apache_beam[gcp]`, `xarray>=2025.7`, `zarr`, `gcsfs`, `xarray-beam`, `jax[cpu]`, `arch`, `numpy>=2.1.3`.
- APIs: `data_loaders/xarray_loaders.py` (`PredictionsFromXarray` L187, `TargetsFromXarray` L236, `ClimatologyFromXarray`, `PersistenceFromXarray`; `rename_dimensions='ecmwf'` maps time→init_time, prediction_timedelta→lead_time / time→valid_time); `weighting.py` `GridAreaWeighting` (L92, grid-agnostic, clamps outer bounds at ±90°); `metrics/deterministic.py` `RMSE` L312, `MSE`, `ACC` L374, `WindVectorRMSE`; `aggregation.py` `Aggregator(reduce_dims, weigh_by, bin_by)` L269; `time_chunks.py` `TimeChunks`; `beam_pipeline.py` `define_pipeline`; `binning.py` `Regions`, `LatitudeBins`, `LandSea`; `interpolations.py`; `metrics/spatial.py` FSS; `statistical_inference/` bootstrap/t-test. Runner: `--runner=DirectRunner` locally.
- 1.40625° / pole-free 128-lat grids handled correctly; predictions and targets must share a grid (no auto-regrid).
- WB2 GCS zarr paths: `gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr`; `…/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr` (1.5°); `…/1959-2022-6h-64x32_equiangular_conservative.zarr`; `…/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr`; climatology `gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr`. Stormer's `download_wb2.py` reads `gs://weatherbench2/datasets/era5/` + file. GCS bulk transfer from this machine: 0.1–0.5 MB/s (unusable).

## 5. torch-harmonics
- Local HEAD `4ac8ed3` (2026-09-18) = `v0.9.3b1 (unreleased)`; PyPI 0.9.2 (2026-08-12), BSD-3.
- API: `RealSHT(nlat, nlon, lmax=None, mmax=None, grid="equiangular", norm="ortho", csphase=True)`, `InverseRealSHT`, `RealVectorSHT`, `InverseRealVectorSHT`. Grids: `equiangular` (Clenshaw–Curtis), `equiangular-trapezoidal`, `legendre-gauss`, `lobatto`.
- `precompute_latitudes` returns **colatitudes in radians** (float64, lru_cached): `lats = flip(arccos(xlg))`.
- **Equiangular includes both poles**: nlat=128 → colat 0°, 1.4173°, …, 180° (spacing 180/127 = 1.4173°), NOT Stormer's pole-free 1.40625° cell centres. No pole-free equiangular option; `legendre-gauss` is pole-free but non-uniform. Longitudes match exactly. Mitigations: interpolate to a 129-lat pole-inclusive grid, or conservative regrid to legendre-gauss, or accept quadrature error (document it).
- v0.9.3b1 changes: theta_cutoff defaults derived from grid spacing; bilinear-spherical resampling fix; legpoly tables re-keyed on (nlat, grid); trapezoidal weights float64.
- Bare source tree does not import (`attention_helpers` extension); use `pip install -e .` or PyPI.

## 6. flash-linear-attention
- Local HEAD `864a87f` (`__version__` 0.6.0 dev); PyPI `flash-linear-attention` / `fla-core` 0.5.2 (2026-07-27), MIT.
- `fla/layers/gated_deltanet.py` `class GatedDeltaNet` (L30); kernels `fla/ops/gated_delta_rule/{chunk,fused_recurrent,naive,…}.py`. Pure-torch `naive_recurrent_gated_delta_rule` / `naive_chunk_gated_delta_rule` exist (fp32, Python loop over T); the `GatedDeltaNet` layer itself requires Triton (`mode in ['chunk','fused_recurrent']`, `ShortConvolution`, `FusedRMSNormGated`). Since v0.5 torch/triton are not base deps; extras `[cuda]` (torch>=2.7, triton>=3.3), `[cpu]`, etc.

## 7. Neighbour-method code
- CLAW 2609.12278: no code. CCM-LoRA: no arXiv record and no GitHub repo found (only the ACL Anthology entry from survey B). Spectral-Target JEPA 2609.04264: no code. EPM-JEPA 2606.12979: no code. **Flow-JEPA 2608.29029: code at https://github.com/HuoYanchen/Flow-JEPA (main `e9172e7`, 2026-08-27, MIT; LICENSE header still credits Lucas Maes).**
- Already-cloned neighbours (HEADs match remote): CoMoL `011306a` (no license); DISeL `b6e543a` (Apache-2.0; paper "coming soon"); PEAR `bb00e7b` (MIT, needs Gurobi); geps `e9a8652` (no license); Solver-in-the-Loop `f514fcf` (MIT); sg-jepa `1b7794b` (MIT; no arXiv ID, self-cites as software); le-wm `8edfeb3` (MIT); StreamTTT `96aed21` (Apache-2.0, default branch master); csubich/graphcast branch `amse` `6ed80d8` (default branch is `graphcast_train`, so pin `amse` explicitly); peft `50a277e` (0.21.1.dev0).

## 8. ERA5 access from this cluster (no GitHub raw, no usable GCS)
Throughput: ModelScope ~7.2 MB/s > hf-mirror ~2.5 MB/s >> GCS 0.1–0.5 MB/s. Reachable: CDS, TUM, AWS `s3://nsf-ncar-era5` (anonymous listing works), Planetary Computer.
1. **hf-mirror datasets**: `JleeOfficial/ERA5-240x121-1979-2018` — 591.43 GB, 1.5° (240×121, WB2 grid), 6-hourly, 13 pressure levels, years 1979–2019, 13 variables + constants/norm stats, npy `(1460,240,121)` / `(1460,13,240,121)`; **no license declared**. `JleeOfficial/ERA5-64x32-1979-2015` — 38.66 GB, 5.625°, same layout. `thainamhoang/era5-climate-learn` (cc-by-4.0) 1.40625° but only 3 surface variables. `jasonjewik/climate-learn` (cc-by-4.0) 5.625°/2.8125° multi-variable shards. `TornikeO/era5-5.625deg` raw WB1 nc, few variables.
2. **TUM WeatherBench-1**: live; download needs a real GET (HEAD/PROPFIND return 401): `wget "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F1.40625deg%2Fgeopotential&files=geopotential_1.40625deg.zip"`. 1.40625° sizes: t2m 35.5 GB, z 344 GB, t 435 GB, q 370 GB (hourly, 1979–2018). 5.625° `all_5.625deg.zip` 267.66 GB. **No mean_sea_level_pressure in WB1** → insufficient for Stormer's 69 channels.
3. **ModelScope**: `OneScience/ERA5` CC-BY-4.0, 0.25°, 243 vars, `data/{year}.h5` 5.05 GB each — but each file holds only T=5 time steps (verified separately: sample set). `LwojvzeL/ERA5_5p625` partial (3 surface vars). `zhangminglang/ERA5_1p5deg_V2` only t/z. `hhs2000/WeatherBench` empty.
4. Others: CDS (needs account; `cdsapi` 0.7.7 on Tsinghua mirror); ECMWF open data = forecasts only; AWS `s3://era5-pds` AccessDenied; AWS `s3://nsf-ncar-era5` anonymous OK (native 0.25° GRIB/netCDF); Planetary Computer STAC `era5-pds`; OpenDataLab / BAAI UNVERIFIED.
Recommendation: (a) `JleeOfficial/ERA5-240x121-1979-2018` via hf-mirror + Stormer's `regrid_wb2.py` (the authors' own path from WB2 240×121), subject to variable-set check and license caveat; (b) TUM 1.40625° only for diagnostics; (c) `JleeOfficial/ERA5-64x32` for quick iteration; (d) CDS or `nsf-ncar-era5` for faithful 0.25° subsets.

## Things that change the plan
1. torch-harmonics cannot represent the 128×256 pole-free grid exactly; choose a mitigation before spectral diagnostics.
2. Stormer requires xformers; budget an SDPA rewrite with layout change.
3. Aurora's LoRA touches only Swin attention qkv/proj; its per-step `lora_mode="all"` driven by `Batch.metadata.rollout_step` is the design to compare against. Aurora is not usable as a coarse second backbone.
4. WeatherBench-X is git-install only with heavy GCS/beam deps; handles 128×256 correctly.
5. No public 1.40625° ERA5 with all 69 variables; realistic path = 1.5° mirror + `regrid_wb2.py`.
6. GraphCast weights now CC BY 4.0; `google-deepmind/graphcast` redirects to `weathernext`.
