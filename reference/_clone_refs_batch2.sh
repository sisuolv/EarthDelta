#!/usr/bin/env bash
# Second batch of reference repos recommended by the literature survey (2026-09-19).
set -u
REF=/mnt/afs/260010168/EarthDelta/reference
LOG="$REF/_clone_log_batch2.txt"
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
# JEPA family (weather+intervention; SIGReg)
clone_one Gen-Verse/JEPA-Anything        JEPA-Anything
clone_one rbalestr-lab/lejepa            lejepa
# adapter bank / coefficient selection baselines
clone_one SakanaAI/text-to-lora          text-to-lora
clone_one sail-sg/lorahub                lorahub
clone_one EnnengYang/AdaMerging          AdaMerging
clone_one tanganke/fusion_bench          fusion_bench
# PDE context adaptation baselines
clone_one yuan-yin/CoDA                  CoDA
clone_one LouisSerrano/zebra             zebra
# adapter featurization / in-context hypernetwork
clone_one xiaolonghan2000/Weight2Token   Weight2Token
clone_one MuLabPKU/SHINE                 SHINE
echo "[done] $(date -u)" | tee -a "$LOG"
