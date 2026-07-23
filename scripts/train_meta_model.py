"""Train the Stage-1 meta-model from Stage-0 tuples (algorithm_plan Sec. 2).

Reads the ``(sketch, operator, useful)`` tuples produced by
``scripts.run_stage0`` and fits one usefulness classifier per operator, saving
the bundled ``MetaModel`` artifact used by the online Stage-4 filter. Prints a
leave-datasets-out evaluation report.

Example:
    python -m scripts.train_meta_model
    python -m scripts.train_meta_model --tuples results/meta/stage0_tuples.jsonl
"""

from __future__ import annotations

import argparse

from afe.meta.stage1 import DEFAULT_MODEL_PATH, DEFAULT_TUPLES_PATH, train_meta_model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tuples", default=str(DEFAULT_TUPLES_PATH),
                   help=f"Stage-0 tuples JSONL (default: {DEFAULT_TUPLES_PATH})")
    p.add_argument("--out", default=str(DEFAULT_MODEL_PATH),
                   help=f"output model path (default: {DEFAULT_MODEL_PATH})")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    _, report = train_meta_model(tuples_path=args.tuples, out_path=args.out,
                                 seed=args.seed, verbose=True)
    print(f"\nsaved meta-model -> {report['model_path']}")


if __name__ == "__main__":
    main()
