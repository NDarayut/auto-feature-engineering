"""Stage 0 -- offline RL label generation (algorithm_plan Sec. 2, Stage 0).

Runs the Double DQN over each corpus dataset's Feature Transformation Graph and
emits the ``(meta-features, operator, usefulness)`` tuples that Stage 1 trains
on. This is the expensive, one-time step: it runs once per corpus dataset, never
at online usage time.

Every transition the environment records during training *is* a labeled tuple
(the DQN's exploration is what makes the search thorough rather than a single
fixed pass over the operator library), so we simply drain
``env.transitions`` after training. Tuples are de-duplicated per dataset by
(rounded-state, operator) keeping the best observed delta, so a feature the
agent revisits many times contributes one clean label, not hundreds.

Output: one JSONL row per tuple, streamed to disk so a long corpus run is
resumable and never has to hold every dataset's tuples in memory at once.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .corpus_data import CorpusDataset, load_corpus_dataset, load_corpus_manifest
from .ddqn import DoubleDQN
from .environment import FTGEnvironment
from .meta_features import SKETCH_DIM

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "meta"
DEFAULT_TUPLES_PATH = RESULTS_DIR / "stage0_tuples.jsonl"


def run_rl_search(
    dataset: CorpusDataset, episodes: int = 60, max_depth: int = 3,
    seed: int = 0, eval_rows: int = 2000, max_features: int = 50,
    verbose: bool = False,
) -> list[dict]:
    """Train the DQN on one dataset's FTG; return de-duplicated labeled tuples."""
    env = FTGEnvironment(dataset, max_depth=max_depth, eval_rows=eval_rows,
                         max_features=max_features, seed=seed)
    agent = DoubleDQN(state_dim=env.state_dim, n_actions=env.n_actions, seed=seed)

    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
        agent.decay_epsilon()
        if verbose and (ep + 1) % 20 == 0:
            useful = sum(t.useful for t in env.transitions)
            print(f"    ep {ep + 1:3d}/{episodes}  transitions={len(env.transitions)}"
                  f"  useful={useful}  eps={agent.eps:.2f}")

    return _dedup_tuples(env, dataset)


def _dedup_tuples(env: FTGEnvironment, dataset: CorpusDataset) -> list[dict]:
    """Collapse repeated (state, operator) visits to one best-delta label."""
    best: dict[tuple, dict] = {}
    for t in env.transitions:
        key = (tuple(np.round(t.feature_sketch, 4)), t.operator)
        row = {
            "did": dataset.did,
            "name": dataset.name,
            "task": dataset.task,
            "operator": t.operator,
            "action": t.action,
            "sketch": [float(v) for v in t.feature_sketch],
            "delta": t.delta,
            "useful": t.useful,
            "depth": t.depth,
        }
        if key not in best or t.delta > best[key]["delta"]:
            best[key] = row
    return list(best.values())


def iter_corpus_datasets(
    limit: int | None = None, use_cache: bool = True, max_rows: int | None = 20_000,
) -> Iterator[CorpusDataset]:
    """Yield prepped corpus datasets from the frozen manifest (skipping failures)."""
    entries = load_corpus_manifest()
    if limit is not None:
        entries = entries[:limit]
    for e in entries:
        try:
            ds = load_corpus_dataset(e["did"], e["name"], e["task"],
                                     use_cache=use_cache, max_rows=max_rows)
        except Exception as exc:  # network/parse failure -> skip, keep going
            print(f"  ! skip did={e['did']} ({e['name']}): {exc}")
            continue
        if ds is None:
            print(f"  ! skip did={e['did']} ({e['name']}): unusable after prep")
            continue
        yield ds


def generate_labels(
    limit: int | None = None, episodes: int = 60, max_depth: int = 3,
    seed: int = 0, out_path: str | Path | None = None, use_cache: bool = True,
    max_rows: int | None = 20_000, max_features: int = 50, verbose: bool = True,
) -> tuple[int, Path]:
    """Run Stage-0 RL search across the corpus, streaming tuples to JSONL.

    Returns ``(n_tuples_written, out_path)``. Resumable by dataset: any ``did``
    already present in an existing output file is skipped.
    """
    out_path = Path(out_path) if out_path else DEFAULT_TUPLES_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_dids = _completed_dids(out_path)

    written = 0
    with out_path.open("a") as fh:
        for ds in iter_corpus_datasets(limit=limit, use_cache=use_cache,
                                       max_rows=max_rows):
            if ds.did in done_dids:
                if verbose:
                    print(f"  = skip did={ds.did} ({ds.name}): already in output")
                continue
            t0 = time.time()
            if verbose:
                print(f"  * did={ds.did} ({ds.name}) task={ds.task} "
                      f"n={len(ds.X)} p={ds.X.shape[1]}")
            rows = run_rl_search(ds, episodes=episodes, max_depth=max_depth,
                                 seed=seed, max_features=max_features,
                                 verbose=verbose)
            for row in rows:
                fh.write(json.dumps(row) + "\n")
            fh.flush()
            written += len(rows)
            if verbose:
                useful = sum(r["useful"] for r in rows)
                print(f"    -> {len(rows)} tuples ({useful} useful) "
                      f"in {time.time() - t0:.1f}s")
    return written, out_path


def _completed_dids(out_path: Path) -> set[int]:
    if not out_path.exists():
        return set()
    dids: set[int] = set()
    for line in out_path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                dids.add(int(json.loads(line)["did"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return dids


# Convenience for tests / sanity: confirm the sketch dim matches the tuples.
EXPECTED_SKETCH_DIM = SKETCH_DIM
