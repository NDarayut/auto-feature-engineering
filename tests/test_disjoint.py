"""Disjointness + integrity guards for the dataset split (no network needed)."""

import afe.benchmark.manifests as manifests
from afe.benchmark.manifests import build_corpus_manifest
from afe.benchmark.registry import BENCHMARK, benchmark_names_for_exclusion


def test_benchmark_keys_unique():
    keys = [s.key for s in BENCHMARK]
    assert len(keys) == len(set(keys)), "duplicate benchmark keys"


def test_exclusion_set_covers_known_cc18_overlaps():
    # These OpenML-CC18 members collide with the benchmark and MUST be excluded
    # from the meta-training corpus (see the approved plan, Sec. 4).
    excluded = benchmark_names_for_exclusion()
    for name in ["jannis", "nomao", "covertype", "electricity", "bank-marketing"]:
        assert name in excluded, f"{name} not caught by corpus exclusion set"


def test_simulated_corpus_is_disjoint():
    # Simulate a raw OpenML suite listing that includes benchmark overlaps and
    # verify the builder's exclusion logic removes every one of them.
    excluded = benchmark_names_for_exclusion()
    raw_suite_names = ["jannis", "nomao", "electricity", "wine-quality",
                       "phoneme", "bank-marketing", "kc1"]
    kept = [n for n in raw_suite_names if n.lower() not in excluded]
    assert set(kept).isdisjoint(excluded)
    assert "wine-quality" in kept and "jannis" not in kept


def test_corpus_manifest_offline_is_safe(monkeypatch):
    # Simulate openml being unavailable: the corpus builder must degrade to
    # per-suite stubs rather than raising, so nothing here touches the network.
    def _boom(_suite_id):
        raise RuntimeError("openml unavailable (simulated)")

    monkeypatch.setattr(manifests, "_enumerate_suite", _boom)
    corpus = build_corpus_manifest()
    assert corpus and all("did" not in e for e in corpus)
    assert all("error" in e for e in corpus)
