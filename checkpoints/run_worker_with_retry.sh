#!/bin/bash
# Restart a pull_wb2 worker on crash (e.g. transient FSTimeoutError opening the
# source zarr store under concurrent load) since open_wb2_zarr() itself has no
# retry, unlike the per-batch fetch loop. Marker-file resume makes restarts safe.
WORKER_ID="$1"
LOG="$2"
MAX_ATTEMPTS=15
ATTEMPT=0
cd /mnt/afs/260010168/EarthDelta
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
  ATTEMPT=$((ATTEMPT+1))
  echo "=== attempt $ATTEMPT/$MAX_ATTEMPTS at $(date -u +%FT%TZ) ===" >> "$LOG"
  python3 -m earthdelta.data.pull_wb2 --year 2020 --worker-id "$WORKER_ID" --https >> "$LOG" 2>&1
  RC=$?
  if [ $RC -eq 0 ]; then
    echo "=== worker $WORKER_ID finished successfully at $(date -u +%FT%TZ) ===" >> "$LOG"
    exit 0
  fi
  echo "=== worker $WORKER_ID exited $RC, retrying in 30s ===" >> "$LOG"
  sleep 30
done
echo "=== worker $WORKER_ID gave up after $MAX_ATTEMPTS attempts ===" >> "$LOG"
exit 1
