# `scripts/` — production entrypoints

Repeatable CLIs that drive the `afe` pipeline. Each is documented in full in
`docs/benchmark_guide.md` (benchmark) or `afe/meta/README.md` (meta-learning).
One-off / exploratory utilities live separately in `../dev/`.

| Script | Purpose |
|---|---|
| `run_benchmark.py` | Run AutoFE methods against the benchmark suite under a compute budget. |
| `report_benchmark.py` | Aggregate a benchmark run's JSONL into the `draft_plan.md` §6 report. |
| `benchmark_ctl.sh` | Start/monitor/stop a long `run_benchmark.py` run in the background. |
| `run_stage0.py` | MF-OpenFE Stage 0 — RL label generation across the meta-training corpus. |
| `train_meta_model.py` | MF-OpenFE Stage 1 — train the meta-model from Stage 0's tuples. |
