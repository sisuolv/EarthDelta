#!/usr/bin/env bash
# Fourth batch: from survey E (tooling): preferred second backbone + Flow-JEPA official code.
set -u
REF=/mnt/afs/260010168/EarthDelta/reference
LOG="$REF/_clone_log_batch4.txt"
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
clone_one INRIA/geoarches          geoarches      # ArchesWeather-M: 1.5deg, 340MB, BSD-3 code+weights, nn.Linear-dense -> preferred second backbone
clone_one HuoYanchen/Flow-JEPA     Flow-JEPA      # official Flow-JEPA code (MIT)
clone_one ai2cm/ace                ace            # ACE2-ERA5: 1deg, Apache-2.0 code+weights (runner-up second backbone)
echo "[done] $(date -u)" | tee -a "$LOG"
