"""Run Stage-0 RL label generation across the meta-training corpus.

Streams ``(sketch, operator, useful)`` tuples to a JSONL file, resumable by
dataset. This is the expensive offline step (algorithm_plan Sec. 2, Stage 0);
run it once, then train the meta-model with ``scripts.train_meta_model``.

Examples:
    python -m scripts.run_stage0 --limit 5 --episodes 40      # quick corpus slice
    python -m scripts.run_stage0 --episodes 80                # full manifest
"""

from __future__ import annotations

import argparse

from afe.meta.stage0 import DEFAULT_TUPLES_PATH, generate_labels


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N corpus datasets (default: all)")
    p.add_argument("--episodes", type=int, default=60,
                   help="RL episodes per dataset (default: 60)")
    p.add_argument("--max-depth", type=int, default=3,
                   help="max transformation chain length per feature (default: 3)")
    p.add_argument("--max-rows", type=int, default=20_000,
                   help="row subsample cap per dataset (default: 20000)")
    p.add_argument("--max-features", type=int, default=50,
                   help="base-feature cap per dataset; wider tables keep the "
                        "top-K target-associated columns (default: 50)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(DEFAULT_TUPLES_PATH),
                   help=f"output JSONL (default: {DEFAULT_TUPLES_PATH})")
    p.add_argument("--no-cache", action="store_true",
                   help="refetch corpus datasets instead of using the parquet cache")
    args = p.parse_args()

    n, out = generate_labels(
        limit=args.limit, episodes=args.episodes, max_depth=args.max_depth,
        seed=args.seed, out_path=args.out, use_cache=not args.no_cache,
        max_rows=args.max_rows, max_features=args.max_features, verbose=True)
    print(f"\nwrote {n} new tuples -> {out}")


if __name__ == "__main__":
    main()
