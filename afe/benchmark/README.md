# `afe.benchmark`

Two things live here:

1. **The research harness** — `registry.py`, `download.py`, `manifests.py`,
   `splits.py`, `eval_data.py`, `models.py`, `benchmark.py` — the frozen
   22-dataset suite this project used to build and validate MF-OpenFE
   against. Full flow: [`docs/benchmark_guide.md`](../../docs/benchmark_guide.md).
2. **`compare()`** (`_compare.py`) — a standalone, algorithm-agnostic API for
   benchmarking *any* AutoFE method. It has no built-in notion of "baseline"
   or "OpenFE" or any other named method — you pass in whatever you want
   compared.

## `compare()`

Plug in your own algorithm — a plain function or a class — no registration,
no editing library source:

```python
from afe.benchmark import compare

# Plain function: (X_train, y_train, X_test, task) -> (X_train_new, X_test_new)
def my_method(X_train, y_train, X_test, task):
    ...
    return X_train_new, X_test_new

# Or a class, same shape as afe.methods.AutoFEMethod:
class MyOtherMethod:
    name = "my-other-method"
    def fit_transform(self, X_train, y_train, task): ...
    def transform(self, X_test): ...

result = compare(
    methods=[my_method, MyOtherMethod],
    datasets=["german-credit"],              # built-in suite, optional
    custom_datasets={"my-data": (X, y)},     # your own data, optional
)
print(result)              # score table, no forced "baseline" column
df = result.to_frame()     # pandas DataFrame for further analysis
```

At least one of `datasets=`/`custom_datasets=` is required. Each dataset
gets a single fixed-seed 80/20 split (same protocol as the rest of
`afe.benchmark`), reused identically across every method being compared.

This is symmetric by design: baseline/OpenFE/Featuretools/Autofeat/MF-OpenFE
are not privileged in any way here — to include one, import it and pass it
in `methods=` exactly like your own algorithm:

```python
from afe.methods import OpenFEMethod, BaselineMethod
from afe import MFOpenFE

compare(methods=[my_method, BaselineMethod, OpenFEMethod, MFOpenFE],
        datasets=["german-credit"])
```

### Execution mode

Runs **in-process by default** — works regardless of where your method is
defined (script, notebook, REPL), with no timeout. Pass `budget_seconds=` to
opt into subprocess isolation + a hard wall-clock budget per (dataset,
method) pair (matches `afe.benchmark.benchmark`'s research-harness
guarantees) — this requires your method to be defined in a real importable
module, since it has to be sent across a process boundary.

### Persistence & resuming

Pass `out_path=` to persist rows as JSONL and make a long comparison
resumable (skips `(dataset, method)` pairs already present):

```python
compare(methods=[...], datasets=[...], out_path="results/my_comparison.jsonl")
```

## Tests
```bash
pytest tests/test_compare.py -q
```
