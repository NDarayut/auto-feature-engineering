"""Offline tests for the MF-OpenFE meta-learning pipeline (afe.meta).

No network: a small synthetic CorpusDataset drives the FTG + DQN + Stage-0/1
flow, so these run in CI without touching OpenML.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afe.meta import ddqn, operators
from afe.meta.corpus_data import CorpusDataset
from afe.meta.environment import FTGEnvironment
from afe.meta.meta_features import SKETCH_DIM, feature_sketch
from afe.meta.stage0 import run_rl_search
from afe.meta.stage1 import MetaModel, train_meta_model


def _synthetic(task: str, n: int = 400, seed: int = 0) -> CorpusDataset:
    rng = np.random.RandomState(seed)
    x0 = rng.randn(n)
    x1 = rng.exponential(size=n)
    x2 = rng.randint(0, 5, size=n).astype(float)
    X = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2})
    if task == "regression":
        y = 2.0 * np.log1p(x1) + 0.5 * x0 + rng.randn(n) * 0.1
    else:
        y = (np.log1p(x1) + 0.3 * x0 > 0.7).astype(float)
    return CorpusDataset(did=1, name="synth", task=task, X=X, y=y,
                         feature_names=("x0", "x1", "x2"))


def test_operators_finite_and_shape():
    x = np.array([-2.0, 0.0, 1.0, 4.0, np.nan])
    for name in operators.OPERATOR_NAMES:
        out = operators.apply_operator(name, x)
        assert out.shape == x.shape
        assert not np.isinf(out).any()  # inf must be mapped to nan


def test_feature_sketch_fixed_length():
    x = np.random.RandomState(0).randn(200)
    y = (x > 0).astype(float)
    vec = feature_sketch(x, y, "classification")
    assert vec.shape == (SKETCH_DIM,)
    assert np.isfinite(vec).all()


def test_ddqn_learns_on_bandit():
    # Degenerate 1-step MDP: action 2 always best -> agent should prefer it.
    rng = np.random.RandomState(0)
    agent = ddqn.DoubleDQN(state_dim=4, n_actions=3, seed=0, eps_end=0.0)
    state = np.ones(4)
    for _ in range(400):
        a = agent.act(state)
        reward = 1.0 if a == 2 else 0.0
        agent.remember(state, a, reward, state, True)
        agent.learn()
        agent.decay_epsilon()
    assert agent.act(state, greedy=True) == 2


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_environment_step_records_transitions(task):
    env = FTGEnvironment(_synthetic(task), max_depth=3, seed=0)
    state = env.reset(feature_index=1)
    assert state.shape == (SKETCH_DIM,)
    total_steps = 0
    done = False
    while not done:
        _, reward, done, info = env.step(1)
        assert np.isfinite(reward)
        total_steps += 1
    assert total_steps <= 3
    assert len(env.transitions) == total_steps


def test_run_rl_search_produces_labeled_tuples():
    rows = run_rl_search(_synthetic("classification"), episodes=15, seed=0)
    assert rows, "RL search should emit at least one tuple"
    r = rows[0]
    assert set(r) >= {"did", "operator", "sketch", "delta", "useful"}
    assert len(r["sketch"]) == SKETCH_DIM
    assert r["operator"] in operators.OPERATOR_NAMES


def test_stage1_trains_and_predicts(tmp_path):
    # Build a tiny two-dataset tuple file so grouped holdout can run.
    import json

    rows = []
    for did, task in [(1, "classification"), (2, "regression")]:
        rows += run_rl_search(_synthetic(task, seed=did), episodes=15, seed=did)
        for r in rows[-1:]:
            r["did"] = did
    # Force distinct dids on every row (synthetic helper hardcodes did=1).
    tuples = tmp_path / "tuples.jsonl"
    with tuples.open("w") as fh:
        for i, r in enumerate(rows):
            r["did"] = 1 if i < len(rows) // 2 else 2
            fh.write(json.dumps(r) + "\n")

    model, report = train_meta_model(tuples_path=tuples,
                                     out_path=tmp_path / "m.pkl", verbose=False)
    assert report["n_tuples"] == len(rows)
    assert report["sketch_dim"] == SKETCH_DIM

    reloaded = MetaModel.load(tmp_path / "m.pkl")
    sketch = np.array(rows[0]["sketch"])
    p = reloaded.score(rows[0]["operator"], sketch)
    assert 0.0 <= p <= 1.0
