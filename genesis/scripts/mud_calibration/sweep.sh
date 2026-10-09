#!/bin/bash
# usage: scripts/mud_calibration/sweep.sh <list.txt>
# Each line is "name|<mud json>". Runs sequentially with the active python and skips names that
# already have results/<name>.json, so an interrupted sweep resumes where it stopped.
cd "$(dirname "$0")"
while IFS='|' read -r name mud; do
  [ -z "$name" ] && continue
  [ -f "results/$name.json" ] && continue
  python -W ignore run_one.py "{\"name\":\"$name\",\"vx\":0.35,\"mud\":$mud}" "results/$name.json" > "results/$name.log" 2>&1
  echo "$name exit=$?"
done < "$1"
echo SWEEP_DONE
