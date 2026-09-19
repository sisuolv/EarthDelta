#!/usr/bin/env bash
# Third batch: repos recommended by surveys C (weather adaptation) and D (response prediction / DFL), 2026-09-19.
set -u
REF=/mnt/afs/260010168/EarthDelta/reference
LOG="$REF/_clone_log_batch3.txt"
: > "$LOG"
PROXIES=("" "https://gh-proxy.com/" "https://ghfast.top/" "https://ghproxy.net/")
export GIT_TERMINAL_PROMPT=0
clone_one() {
  local repo="$1" dir="$2" branch="${3:-}" depth="${4:-1}"
  local dest="$REF/$dir"
  if [ -d "$dest/.git" ] || [ -f "$dest/_FROM_ZIP" ]; then echo "[skip] $dir exists" | tee -a "$LOG"; return 0; fi
  rm -rf "$dest"
  local args=(-q --depth "$depth"); [ -n "$branch" ] && args+=(--branch "$branch")
  for p in "${PROXIES[@]}"; do
    echo "[try] $repo via '${p:-direct}'" | tee -a "$LOG"
    if timeout 900 git clone "${args[@]}" "${p}https://github.com/${repo}.git" "$dest" 2>>"$LOG"; then
      git -C "$dest" remote set-url origin "https://github.com/${repo}.git"
      echo "[ok] $repo -> $dir @ $(git -C "$dest" rev-parse --short HEAD) $(git -C "$dest" log -1 --format=%cs)" | tee -a "$LOG"; return 0
    fi
    rm -rf "$dest"
  done
  echo "[FAIL] $repo" | tee -a "$LOG"; return 1
}
# weather-domain adaptation / perturbation / correction
clone_one ShileiCao/WeatherPEFT              WeatherPEFT
clone_one MeteoSwiss/ai-models-ensembles     ai-models-ensembles     # SPW: stochastically perturbed weights of frozen MLWMs
clone_one akhtarvision/weather-regional      weather-regional        # LoRA regional adaptation (MENA)
clone_one Fifthky/ORCA                       ORCA                    # black-box online adaptation on "context of errors"
clone_one BorealisAI/PETSA                   PETSA                   # low-rank adapters + gating at test time
clone_one tum-pbs/INC                        INC                     # indirect neural corrector
clone_one ddrous/ncflow                      ncflow                  # neural context flows
clone_one yuan-yin/LEADS                     LEADS
# decision-focused learning / QP layers / amortized optimization
clone_one khalil-research/PyEPO              PyEPO
clone_one cvxgrp/cvxpylayers                 cvxpylayers
clone_one locuslab/qpth                      qpth
clone_one facebookresearch/amortized-optimization-tutorial amortized-optimization-tutorial
# edit interference / merging baselines
clone_one prateeky2806/ties-merging          ties-merging
clone_one mlfoundations/task_vectors         task_vectors
clone_one arcee-ai/mergekit                  mergekit
echo "[done] $(date -u)" | tee -a "$LOG"
