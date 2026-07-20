# Using `scripts/benchmark_ctl.sh`

A quick, task-oriented guide to running and monitoring the benchmark
yourself. For how the benchmark itself works (splits, encoding, methods,
models), see `docs/benchmark_guide.md` — this doc is just the control
script's interface.

## What it's for

The benchmark run is long (many hours across all 22 datasets × 4 methods)
and needs to run in the background. `benchmark_ctl.sh` wraps the plumbing
so you don't have to hand-manage `nohup`, PID files, or log redirection —
and it avoids a real pitfall: killing a naively-backgrounded Python process
can leave its `multiprocessing` workers running and consuming memory after
the "main" process is gone. This script starts everything in one process
group and stops the whole group at once.

## Setup

Run all commands from the repo root, with the datasets already fetched (see
`docs/dataset_setup.md`) and `.venv` present. The script activates
`.venv` itself if it finds one, so you don't need to `source` it first.

```bash
cd auto-feature-engineering
chmod +x scripts/benchmark_ctl.sh   # first time only
```

## Commands

### `start` — launch a run in the background

```bash
scripts/benchmark_ctl.sh start -- [run_benchmark.py options]
```

Everything after `--` is passed straight through to
`scripts/run_benchmark.py`. Datasets are always processed **sequentially,
one at a time** — the harness loads a dataset, runs every requested method
against it, then frees that dataset's memory before moving to the next.
Useful options:

| flag | meaning |
|---|---|
| `--budget S` | per (dataset, method) feature-generation time limit in seconds (default 300) |
| `--methods ...` | which AutoFE methods to run, any of `baseline openfe featuretools autofeat` |
| `--models ...` | which model families to evaluate with, any of `tree linear knn` (default: all three) |
| `--datasets ...` | which dataset keys to run (default: all 22, smallest first) |
| `--no-resume` | re-run pairs even if already in the results file (default is to skip them) |

Examples:

```bash
# A fast first pass: all methods, tree model only
scripts/benchmark_ctl.sh start -- --budget 300 --models tree \
    --methods baseline openfe featuretools autofeat

# Just check two datasets against openfe, with a short budget
scripts/benchmark_ctl.sh start -- --datasets nomao concrete-strength \
    --methods baseline openfe --budget 60
```

If a run is already active, `start` refuses to launch a second one (checked
via `results/benchmark.pid`) — use `status` to see what's running, or `stop`
it first.

### `status` — check progress at any time

```bash
scripts/benchmark_ctl.sh status
```

```
Status: RUNNING (pgid 147395)
Total rows: 482
(dataset, method) pairs touched: 56 / 88 (22 datasets x 4 methods)
Status breakdown:
  ok              459
  timeout         14
  crashed         4
  error           1
```

This works whether or not a run is currently active — it just reads
whatever's in `results/benchmark_results.jsonl` so far. Run it as often as
you like; it doesn't affect the run itself.

**Reading the status breakdown:**
- `ok` — completed normally.
- `timeout` — the method didn't finish generating features within `--budget`
  seconds; recorded as a real result, not an error (a method that can't
  keep up with its budget on a given dataset is itself informative).
- `crashed` — the feature-generation subprocess died unexpectedly (often
  the OS OOM-killer — see below). You can find it with:
  ```bash
  grep '"crashed"' results/benchmark_results.jsonl
  ```
- `error` — the method or model raised a Python exception, caught and
  logged rather than stopping the run.

### `tail` — watch it live

```bash
scripts/benchmark_ctl.sh tail
```

Follows `results/logs/run.log` (model training logs, progress bars, etc.).
`Ctrl-C` only stops *watching* — the run keeps going in the background.

### `stop` — stop it

```bash
scripts/benchmark_ctl.sh stop
```

Kills the entire process group (main script + the current dataset's nested
feature-generation subprocess), not just the top-level process. Progress
already written to `results/benchmark_results.jsonl` is untouched — running
`start` again later picks up where it left off, since finished
`(dataset, method)` pairs are automatically skipped.

### `report` — turn results into a readable summary

```bash
scripts/benchmark_ctl.sh report -- --out results/report.md
# or straight to your terminal:
scripts/benchmark_ctl.sh report
```

Works on whatever's in the results file at that moment — you don't need to
wait for the run to finish. Produces the point-delta-vs-baseline table
described in `docs/benchmark_guide.md` §8 (one value per dataset/method/model,
from the single fixed-seed split — no mean±std or significance marker, since
there's no distribution to compute one over).

## A typical session

```bash
scripts/benchmark_ctl.sh start -- --methods baseline openfe featuretools autofeat

scripts/benchmark_ctl.sh status        # check in whenever
scripts/benchmark_ctl.sh status        # ... again later

scripts/benchmark_ctl.sh stop          # done for now, or need the machine back
scripts/benchmark_ctl.sh start -- --methods baseline openfe featuretools autofeat
    # resumes automatically

scripts/benchmark_ctl.sh report -- --out results/report.md
```

## Memory

`featuretools` and `autofeat` have been observed using anywhere from a few
hundred MB to 20+ GB on the larger datasets (700k+ rows), inherent to those
libraries' own feature-generation internals. The harness no longer runs
datasets concurrently, so this is no longer multiplied by a worker count —
at most one dataset's generation step is in memory at a time, and that
subprocess's memory is released the moment it exits (whether it finishes,
times out, or is killed).

- Check live memory while a run is active with `free -h`, or find the
  heaviest process with `ps aux --sort=-%mem | head`.
- If `status` shows `crashed` rows, that's the kernel OOM-killer taking out
  the generation subprocess for a single dataset/method — not a bug to
  chase in the harness itself. Already-completed pairs are skipped on
  resume, so this costs time on that one pair, not lost work elsewhere.

## Troubleshooting

**"Already running" but you don't think it is.** `results/benchmark.pid`
is stale (e.g. the machine restarted). Check `ps aux | grep run_benchmark`
— if nothing's there, just `rm results/benchmark.pid` and `start` again.

**A run was killed some other way (not via `stop`) and things look messy.**
Check for leftover processes and clean up manually:
```bash
ps aux | grep -E "run_benchmark|multiprocessing.spawn" | grep -v grep
# kill any that are clearly orphaned (ppid no longer the benchmark script)
```
This shouldn't happen if you always use `stop`, but if you `Ctrl-C` a
foreground run or kill the wrong PID by hand, the nested feature-generation
subprocess can be left behind consuming memory.

**Want to change `--budget`/`--methods`/`--models` mid-run.** There's no live
reconfiguration — `stop`, then `start` again with the new flags. Already-
completed pairs are skipped, so nothing is lost.
