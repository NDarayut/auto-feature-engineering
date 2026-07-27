"""CAFEM-style RL search components, backing ``afe.methods.CAFEMMethod``.

Nothing here runs on its own; these are the pieces the CAFEM method drives
per dataset, on the training fold only:

* ``environment`` -- ``FTGEnvironment``, a Feature Transformation Graph over
  one dataset: state = a feature's meta-feature sketch, action = a unary
  operator, reward = the wrapper-model improvement from adding the
  transformed feature.
* ``ddqn`` -- ``DoubleDQN``, the agent that explores that graph.
* ``operators`` -- the unary operator library the graph's edges apply.
* ``meta_features`` -- the LFE-style QSA sketch used as the agent's state,
  plus the univariate target-association score ``afe.benchmark.models``
  reuses for its feature-efficiency metric.
* ``corpus_data`` -- ``CorpusDataset``, the plain (X, y, task) container
  ``FTGEnvironment`` takes as input.
"""

from __future__ import annotations

from .corpus_data import CorpusDataset
from .ddqn import DoubleDQN
from .environment import FTGEnvironment
from .meta_features import feature_sketch
from .operators import apply_operator

__all__ = [
    "CorpusDataset",
    "DoubleDQN",
    "FTGEnvironment",
    "apply_operator",
    "feature_sketch",
]
