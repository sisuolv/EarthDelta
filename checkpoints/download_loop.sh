#!/bin/bash
set -u
cd /mnt/afs/260010168/EarthDelta/checkpoints

download_until_done() {
  local url="$1" out="$2" expect="$3"
  for i in $(seq 1 100); do
    local size=0
    [ -f "$out" ] && size=$(stat -c%s "$out")
    if [ "$size" -eq "$expect" ]; then
      echo "$out: exact expected size $expect reached (attempt $i)"
      return 0
    fi
    if [ "$size" -gt "$expect" ]; then
      echo "$out: size $size exceeds expected $expect, truncating and restarting clean"
      rm -f "$out"
      size=0
    fi
    echo "$out: attempt $i, current size $size / $expect"
    curl -sS -L -C - --connect-timeout 20 --max-time 300 -o "$out" "$url"
    local newsize=0
    [ -f "$out" ] && newsize=$(stat -c%s "$out")
    if [ "$newsize" -gt "$expect" ]; then
      echo "$out: overshoot detected ($newsize > $expect), curl -C resume was not honored; deleting for clean restart"
      rm -f "$out"
    fi
    sleep 2
  done
  echo "$out: FAILED after 100 attempts"
  return 1
}

download_until_done "https://hf-mirror.com/tungnd/stormer/resolve/main/stormer_1.40625_patch_size_4.ckpt" stormer_1.40625_patch_size_4.ckpt 5570407547
echo "ALL_DONE"
