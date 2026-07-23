"""MF-OpenFE: automatic feature engineering, meta-filtered.

Public entrypoint::

    from afe import MFOpenFE

    mfe = MFOpenFE(task="classification", progress=True)
    X_train_fe = mfe.fit_transform(X_train, y_train)
    X_test_fe = mfe.transform(X_test)

Offline meta-model training utilities (``MetaModel``, ``train_meta_model``,
``generate_labels``) are also re-exported for advanced/research use -- see
``afe/meta/README.md``.

The AutoFE benchmark/comparison harness (baseline vs. OpenFE vs. Featuretools
vs. Autofeat vs. MF-OpenFE, plus the frozen dataset registry it runs against)
lives in the separate ``afe.benchmark`` subpackage and is not part of this
top-level surface -- see ``docs/benchmark_guide.md``.
"""

from __future__ import annotations

from .meta import MetaModel, MFOpenFE, generate_labels, train_meta_model

__all__ = [
    "MetaModel",
    "MFOpenFE",
    "generate_labels",
    "train_meta_model",
]
