#!/bin/bash
# Sequentially pull and finalize ERA5 data for multiple years, one at a time
# (to avoid OOM in this memory-constrained cgroup). 2020 is already pulled
# and finalized, so it is skipped here.
YEARS="2015 2016 2017 2018 2019 2021 2022"
MAX_ATTEMPTS=15
TOP_LOG="checkpoints/multi_year_pull.log"
cd /mnt/afs/260010168/EarthDelta
# /root/.local (pip --user) does not survive a CCI container restart; these
# deps live on the persistent AFS mount instead.
export PYTHONPATH=/mnt/afs/260010168/EarthDelta/.pydeps${PYTHONPATH:+:$PYTHONPATH}

run_with_retry() {
  # $1 = log file, $2 = step name (for messages), remaining args = command
  LOG="$1"
  STEP="$2"
  shift 2
  ATTEMPT=0
  while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT+1))
    echo "=== $STEP attempt $ATTEMPT/$MAX_ATTEMPTS at $(date -u +%FT%TZ) ===" >> "$LOG"
    "$@" >> "$LOG" 2>&1
    RC=$?
    if [ $RC -eq 0 ]; then
      echo "=== $STEP finished successfully at $(date -u +%FT%TZ) ===" >> "$LOG"
      return 0
    fi
    echo "=== $STEP exited $RC, retrying in 30s ===" >> "$LOG"
    sleep 30
  done
  echo "=== $STEP gave up after $MAX_ATTEMPTS attempts ===" >> "$LOG"
  return 1
}

for YEAR in $YEARS; do
  LOG="checkpoints/pull_${YEAR}.log"

  run_with_retry "$LOG" "year $YEAR sequential pull" \
    python3 -m earthdelta.data.pull_wb2 --year "$YEAR" --sequential --https
  if [ $? -ne 0 ]; then
    MSG="FATAL: year $YEAR failed at pull, aborting remaining years"
    echo "$MSG" >> "$LOG"
    echo "$MSG" >> "$TOP_LOG"
    exit 1
  fi

  run_with_retry "$LOG" "year $YEAR finalize" \
    python3 -m earthdelta.data.pull_wb2 --year "$YEAR" --finalize
  if [ $? -ne 0 ]; then
    MSG="FATAL: year $YEAR failed at finalize, aborting remaining years"
    echo "$MSG" >> "$LOG"
    echo "$MSG" >> "$TOP_LOG"
    exit 1
  fi
done

echo "=== all years complete at $(date -u +%FT%TZ) ===" >> "$TOP_LOG"
exit 0
