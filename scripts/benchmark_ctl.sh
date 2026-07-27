#!/usr/bin/env bash
# Start, monitor, and stop the benchmark run yourself, without needing a
# second terminal glued to it. Wraps scripts/run_benchmark.py + report_benchmark.py.
#
# Usage:
#   scripts/benchmark_ctl.sh start [-- extra run_benchmark.py args]
#   scripts/benchmark_ctl.sh status
#   scripts/benchmark_ctl.sh tail
#   scripts/benchmark_ctl.sh stop
#   scripts/benchmark_ctl.sh report [-- extra report_benchmark.py args]
#
# Examples:
#   scripts/benchmark_ctl.sh start -- --budget 300
#   scripts/benchmark_ctl.sh start -- --budget 300 --models tree
#   scripts/benchmark_ctl.sh status
#   scripts/benchmark_ctl.sh report -- --out results/report.md

set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS_DIR="results"
LOG_DIR="$RESULTS_DIR/logs"
PID_FILE="$RESULTS_DIR/benchmark.pid"
OUT_FILE="$RESULTS_DIR/benchmark_results.jsonl"
LOG_FILE="$LOG_DIR/run.log"

mkdir -p "$LOG_DIR"

activate_venv() {
  if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  fi
}

is_running() {
  [ -f "$PID_FILE" ] && kill -0 -- "-$(cat "$PID_FILE")" 2>/dev/null
}

cmd_start() {
  if is_running; then
    echo "Already running (pgid $(cat "$PID_FILE")). Use 'status' or 'stop' first."
    exit 1
  fi
  activate_venv
  echo "Starting: python -m scripts.run_benchmark $* --out $OUT_FILE"
  # `set -m` (job control) puts the background job in its own process
  # group, with $! reliably equal to that group's PGID -- nested
  # multiprocessing workers inherit the same group. That lets `stop` kill
  # the whole tree at once with `kill -- -PGID` instead of leaking
  # orphaned subprocesses (this bit us during development -- see
  # the README). `setsid` was tried first and rejected: it
  # forks internally, so the PID bash captures via $! is not reliably the
  # PID of the actual long-running process.
  set -m
  nohup python -m scripts.run_benchmark "$@" --out "$OUT_FILE" \
    > "$LOG_FILE" 2>&1 < /dev/null &
  local pid=$!
  set +m
  echo "$pid" > "$PID_FILE"
  disown
  echo "Started (pgid $pid). Log: $LOG_FILE"
  echo "Check progress any time with: $0 status"
}

cmd_status() {
  if is_running; then
    echo "Status: RUNNING (pgid $(cat "$PID_FILE"))"
  else
    echo "Status: NOT RUNNING"
    [ -f "$PID_FILE" ] && rm -f "$PID_FILE"
  fi
  if [ ! -f "$OUT_FILE" ]; then
    echo "No results yet ($OUT_FILE doesn't exist)."
    return
  fi
  activate_venv
  python - "$OUT_FILE" <<'PYEOF'
import json
import sys
from collections import Counter

path = sys.argv[1]
statuses = Counter()
pairs = set()
n = 0
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        statuses[row["status"]] += 1
        pairs.add((row["key"], row["method"]))
        n += 1

print(f"Total rows: {n}")
print(f"(dataset, method) pairs touched: {len(pairs)} / 88 (22 datasets x 4 methods)")
print("Status breakdown:")
for status, count in sorted(statuses.items(), key=lambda kv: -kv[1]):
    print(f"  {status:15s} {count}")
PYEOF
}

cmd_tail() {
  echo "Following $LOG_FILE (Ctrl-C stops watching; the run keeps going)."
  tail -f "$LOG_FILE"
}

cmd_stop() {
  if is_running; then
    local pid
    pid=$(cat "$PID_FILE")
    echo "Stopping process group $pid ..."
    kill -TERM -- "-$pid" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-$pid" 2>/dev/null || true
  else
    echo "Not running."
  fi
  rm -f "$PID_FILE"
  echo "Stopped. Progress in $OUT_FILE is preserved -- 'start' again later to resume"
  echo "(already-completed (dataset, method) pairs are skipped automatically)."
}

cmd_report() {
  activate_venv
  python -m scripts.report_benchmark "$OUT_FILE" "$@"
}

main() {
  local sub="${1:-}"
  shift || true
  # allow a literal "--" separator before pass-through args, for readability
  if [ "${1:-}" = "--" ]; then shift; fi
  case "$sub" in
    start) cmd_start "$@" ;;
    status) cmd_status ;;
    tail) cmd_tail ;;
    stop) cmd_stop ;;
    report) cmd_report "$@" ;;
    *)
      cat <<EOF
Usage: $0 {start|status|tail|stop|report} [-- extra args]

  start [-- ARGS]   Launch the benchmark run in the background (resumable).
                    Datasets are processed sequentially, one at a time.
                    ARGS are passed through to scripts/run_benchmark.py,
                    e.g. -- --budget 300 --models tree
  status            Show whether it's running + row/status counts so far.
  tail              Follow the live log (Ctrl-C to stop watching).
  stop              Stop the background run (kills the whole process
                    group, not just the top process -- avoids leaking
                    multiprocessing workers).
  report [-- ARGS]  Aggregate results into a report. ARGS passed through
                    to scripts/report_benchmark.py, e.g. -- --out results/report.md
EOF
      exit 1
      ;;
  esac
}

main "$@"
