#!/bin/bash
# Restart the pull_wb2 --sequential process on crash (container restarts,
# transient network errors). Marker-file resume makes restarts safe.
LOG="$1"
MAX_ATTEMPTS=15
ATTEMPT=0
cd /mnt/afs/260010168/EarthDelta
# /root/.local (pip --user) does not survive a CCI container restart; these
# deps live on the persistent AFS mount instead.
export PYTHONPATH=/mnt/afs/260010168/EarthDelta/.pydeps${PYTHONPATH:+:$PYTHONPATH}
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
  ATTEMPT=$((ATTEMPT+1))
  echo "=== attempt $ATTEMPT/$MAX_ATTEMPTS at $(date -u +%FT%TZ) ===" >> "$LOG"
  python3 -m earthdelta.data.pull_wb2 --year 2020 --sequential --https >> "$LOG" 2>&1
  RC=$?
  if [ $RC -eq 0 ]; then
    echo "=== sequential pull finished successfully at $(date -u +%FT%TZ) ===" >> "$LOG"
    exit 0
  fi
  echo "=== sequential pull exited $RC, retrying in 30s ===" >> "$LOG"
  sleep 30
done
echo "=== sequential pull gave up after $MAX_ATTEMPTS attempts ===" >> "$LOG"
exit 1
