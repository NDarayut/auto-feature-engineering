# `dev/` — one-off / exploratory utilities

Manual smoke- and sanity-check scripts used while developing, **not** part of
the production pipeline (that's `../scripts/`). Safe to run repeatedly but
not meant to be scheduled or depended on by other code.

| Script | Purpose |
|---|---|
| `smoke_download.py` | Fetch every benchmark dataset and print its metadata row; fails loudly on a broken source. |
| `parity_check.py` | Raw-feature LightGBM baseline on a few datasets — confirms the split/encode/fold path still works, sanity gate not a full run. |
