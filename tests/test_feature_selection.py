import numpy as np
import polars as pl

from feature_selection import (
    FeatureImportance,
    FeatureSelector,
    RandomForestSelector,
    ThresholdConfig,
    _select_core,
)


def _importance(names, values):
    return [
        FeatureImportance(name=n, importance=v)
        for n, v in sorted(zip(names, values, strict=False), key=lambda x: x[1], reverse=True)
    ]


def test_max_features_hard_cap():
    names = [f"f{i}" for i in range(20)]
    X = np.random.RandomState(0).normal(size=(2000, 20))
    # nearly uniform importance so the cumulative target (0.70) needs many features
    values = np.linspace(0.09, 0.01, 20)
    values /= values.sum()
    result = _select_core(
        names,
        X,
        _importance(names, values),
        ThresholdConfig(
            correlation_threshold=0.99,
            cumulative_importance=0.70,
            max_features=15,
        ),
    )
    assert len(result.features) <= 15


def test_min_features_floor_respected():
    # one dominant feature satisfies the cumulative target immediately; the
    # floor must pad the selection back up to min_features (5)
    names = [f"f{i}" for i in range(8)]
    X = np.random.RandomState(1).normal(size=(2000, 8))
    values = np.array([0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02])
    result = _select_core(
        names,
        X,
        _importance(names, values),
        ThresholdConfig(
            correlation_threshold=0.99,
            cumulative_importance=0.70,
            min_features=5,
            max_features=15,
        ),
    )
    assert len(result.features) == 5


def test_correlation_prune_keeps_higher_ranked():
    # f0 and f1 are perfectly correlated; f0 must survive, f1 must go.
    # min_features=1 keeps the hard floor from masking the correlation prune.
    rng = np.random.RandomState(2)
    base = rng.normal(size=(2000, 1))
    X = np.column_stack([base, base[:, 0], rng.normal(size=2000)])
    names = ["corr_top", "corr_bottom", "independent"]
    values = np.array([0.5, 0.2, 0.3])
    result = _select_core(
        names,
        X,
        _importance(names, values),
        ThresholdConfig(
            correlation_threshold=0.85,
            cumulative_importance=0.90,
            min_features=1,
            max_features=15,
        ),
    )
    assert "corr_top" in result.features
    assert "corr_bottom" not in result.features


def test_correlation_prune_prefers_stable_semantics():
    # `liquidity_high` outranks `liquidity` but is derived; the preferred
    # base feature must win the correlated pair.
    rng = np.random.RandomState(5)
    base = rng.normal(size=(2000, 1))
    X = np.column_stack([base, base[:, 0]])
    names = ["liquidity_high", "liquidity"]
    values = np.array([0.6, 0.4])
    result = _select_core(
        names,
        X,
        _importance(names, values),
        ThresholdConfig(
            correlation_threshold=0.85,
            cumulative_importance=0.99,
            min_features=1,
            max_features=15,
            preferred_features=("liquidity", "market_cap"),
        ),
    )
    assert result.features == ["liquidity"]
    assert "liquidity_high" not in result.features


def test_cap_reason_reports_binding_constraint():
    rng = np.random.RandomState(6)
    X = rng.normal(size=(2000, 30))
    names = [f"f{i}" for i in range(30)]
    values = np.linspace(0.2, 0.01, 30)
    values /= values.sum()

    result = _select_core(
        names,
        X,
        _importance(names, values),
        ThresholdConfig(
            correlation_threshold=0.99,
            cumulative_importance=0.999,
            max_features=10,
        ),
    )
    assert len(result.features) == 10
    assert result.cap_reason == "max_features"


def test_select_from_frame_path_matches_select():
    rng = np.random.RandomState(3)
    n = 3000
    base = rng.normal(size=(n, 1))
    cols = {"a": base[:, 0], "b": base[:, 0]}  # perfect corr pair
    cols.update({f"c{i}": rng.normal(size=n) for i in range(8)})
    cols["label"] = (base[:, 0] > 0).astype(np.int32)
    frame = pl.DataFrame(cols)
    selector = FeatureSelector(threshold_config=ThresholdConfig())
    result = selector.select_from_frame(frame)
    assert len(result.features) <= 15
    # perfectly-correlated pair a/b must collapse to exactly one member
    assert not ({"a", "b"} <= set(result.features))
    assert result.cap_reason in ("cumulative", "max_features", "all")


def test_random_forest_selector_delegates_to_core():
    rng = np.random.RandomState(4)
    X = rng.normal(size=(2000, 8))
    y = (X[:, 0] + 0.5 * rng.normal(size=2000) > 0).astype(int)
    selector = RandomForestSelector(
        n_estimators=20, max_depth=5, min_samples=100, cumulative_threshold=0.70
    )
    result = selector.select_from_arrays(X, y, [f"f{i}" for i in range(8)])
    assert 0 < len(result.features) <= 15
    assert len(result.importance) == 8
