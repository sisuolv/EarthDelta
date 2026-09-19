#!/usr/bin/env bash
# Clone EarthDelta reference repos with fallbacks: direct -> gh-proxy.com -> ghfast.top -> ghproxy.net -> codeload zip
set -u
REF=/mnt/afs/260010168/EarthDelta/reference
LOG="$REF/_clone_log.txt"
: > "$LOG"
PROXIES=("" "https://gh-proxy.com/" "https://ghfast.top/" "https://ghproxy.net/")
export GIT_TERMINAL_PROMPT=0

clone_one() {
  # args: owner/repo  dirname  branch(or "")  depth(or "")
  local repo="$1" dir="$2" branch="${3:-}" depth="${4:-}"
  local dest="$REF/$dir"
  if [ -d "$dest/.git" ] || [ -f "$dest/_FROM_ZIP" ]; then echo "[skip] $dir exists" | tee -a "$LOG"; return 0; fi
  rm -rf "$dest"
  local args=(-q)
  [ -n "$branch" ] && args+=(--branch "$branch")
  [ -n "$depth" ] && args+=(--depth "$depth")
  for p in "${PROXIES[@]}"; do
    local url="${p}https://github.com/${repo}.git"
    echo "[try] $repo via '${p:-direct}'" | tee -a "$LOG"
    if timeout 900 git clone "${args[@]}" "$url" "$dest" 2>>"$LOG"; then
      git -C "$dest" remote set-url origin "https://github.com/${repo}.git"
      echo "[ok] $repo -> $dir @ $(git -C "$dest" rev-parse --short HEAD) $(git -C "$dest" log -1 --format=%cs)" | tee -a "$LOG"
      return 0
    fi
    rm -rf "$dest"
  done
  local ref="HEAD"; [ -n "$branch" ] && ref="refs/heads/${branch}"
  local zipurl="https://codeload.github.com/${repo}/zip/${ref}"
  echo "[try] $repo via codeload zip $zipurl" | tee -a "$LOG"
  if timeout 600 curl -sfL "$zipurl" -o "$REF/_tmp_$dir.zip" && unzip -q "$REF/_tmp_$dir.zip" -d "$REF/_tmp_$dir"; then
    mkdir -p "$dest"; mv "$REF/_tmp_$dir"/*/* "$REF/_tmp_$dir"/*/.[!.]* "$dest" 2>/dev/null; rm -rf "$REF/_tmp_$dir" "$REF/_tmp_$dir.zip"
    echo "zip snapshot from $zipurl on $(date -u +%F)" > "$dest/_FROM_ZIP"
    echo "[ok-zip] $repo -> $dir (no git history)" | tee -a "$LOG"; return 0
  fi
  rm -rf "$REF/_tmp_$dir" "$REF/_tmp_$dir.zip"
  echo "[FAIL] $repo" | tee -a "$LOG"; return 1
}

# --- backbones & data/eval tooling ---
clone_one tung-nd/stormer                     stormer                 ""     ""      # full history: need pinned commit
clone_one microsoft/aurora                    aurora                  ""     1
clone_one microsoft/ClimaX                    ClimaX                  ""     1       # Stormer predecessor; 1.40625deg ERA5 preprocessing
clone_one google-research/weatherbenchX       weatherbenchX           ""     1
clone_one google-research/weatherbench2       weatherbench2           ""     1
clone_one NVIDIA/torch-harmonics              torch-harmonics         ""     1
clone_one fla-org/flash-linear-attention      flash-linear-attention  ""     1
clone_one huggingface/peft                    peft                    ""     1
# --- spectral loss ---
clone_one csubich/graphcast                   graphcast-amse          amse   1
clone_one google-deepmind/graphcast           graphcast               ""     1
# --- method neighbours ---
clone_one lucas-maes/le-wm                    le-wm                   ""     1
clone_one sg-jepa/sg-jepa                     sg-jepa                 ""     1
clone_one zeyun-zhong/StreamTTT               StreamTTT               ""     1
clone_one DCDmllm/CoMoL                       CoMoL                   ""     1
clone_one alizindari/DISeL                    DISeL                   ""     1
clone_one FinJun/PEAR                         PEAR                    ""     1
clone_one tum-pbs/Solver-in-the-Loop          Solver-in-the-Loop      ""     1
clone_one itsakk/geps                         geps                    ""     1

# pin stormer to the commit referenced in the plans, keep main available
if [ -d "$REF/stormer/.git" ]; then
  PIN=58dfee5a6037399a40fefd492bc00421e0c885a8
  if git -C "$REF/stormer" cat-file -e "$PIN^{commit}" 2>/dev/null; then
    AHEAD=$(git -C "$REF/stormer" rev-list --count "$PIN"..HEAD)
    echo "[stormer] main HEAD=$(git -C "$REF/stormer" rev-parse --short HEAD); pinned commit exists; main is $AHEAD commits ahead of pin" | tee -a "$LOG"
    git -C "$REF/stormer" branch -f earthdelta_pin "$PIN" && echo "[stormer] branch earthdelta_pin created at $PIN (checkout left on main)" | tee -a "$LOG"
  else
    echo "[stormer] WARNING pinned commit $PIN NOT found in history" | tee -a "$LOG"
  fi
fi
echo "[done] $(date -u)" | tee -a "$LOG"
