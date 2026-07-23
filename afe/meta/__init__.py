"""MF-OpenFE offline meta-learning pipeline (algorithm_plan.md).

This package implements the *offline, one-time* half of MF-OpenFE:

* Stage 0 -- RL search (CAFEM-style Double DQN over a Feature Transformation
  Graph) that, per historical corpus dataset, discovers which single-feature
  transformations improve a wrapper model and emits
  ``(meta-features, operator, usefulness)`` tuples. (``stage0``)
* Stage 1 -- a supervised meta-model (per-operator RF/GBM) trained on those
  pooled tuples, mapping an LFE-style QSA feature sketch to predicted
  usefulness. This is the only artifact used at online usage time. (``stage1``)

* Stages 3-6 -- the *online* path: every new dataset, every time, runs live.
  OpenFE's own candidate generator, filtered by the trained Stage 1 model,
  then verified + selected by feature importance. This is ``MFOpenFE``
  (``online.py``) -- the public entrypoint, also re-exported as ``afe.MFOpenFE``.

Common entrypoints are re-exported here for convenience::

    from afe.meta import MFOpenFE, MetaModel, train_meta_model, generate_labels

Every submodule remains directly importable too -- this is purely additive.
"""

from __future__ import annotations

from .environment import FTGEnvironment
from .meta_features import feature_sketch
from .online import MFOpenFE
from .stage0 import generate_labels, run_rl_search
from .stage1 import MetaModel, train_meta_model

__all__ = [
    "FTGEnvironment",
    "MFOpenFE",
    "MetaModel",
    "feature_sketch",
    "generate_labels",
    "run_rl_search",
    "train_meta_model",
]
