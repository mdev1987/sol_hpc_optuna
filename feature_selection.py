from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dataset_builder import TrainingDataset


@dataclass(slots=True)
class FeatureImportance:
    name: str
    importance: float


@dataclass(slots=True)
class SelectionResult:
    features: list[str]
    importance: list[FeatureImportance]
    cap_reason: str = ""  # what bound the selection: cumulative | max_features | min_features | all


def _dataset_arrays(dataset: TrainingDataset, names: list[str] | None = None):
    if names is None:
        names = list(dataset.samples[0].features.keys())
    X = np.empty((len(dataset), len(names)), dtype=np.float64)
    y = np.empty(len(dataset), dtype=np.int32)
    for i, s in enumerate(dataset.samples):
        for j, f in enumerate(names):
            X[i, j] = s.features[f]
        y[i] = s.label
    return X, y, names


def _safe_corrcoef(X: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


@dataclass(slots=True)
class ThresholdConfig:
    correlation_threshold: float = 0.85
    variance_threshold: float = 0.01
    min_features: int = 5
    max_features: int = 15
    cumulative_importance: float = 0.70
    # When two features are correlated, prefer the one listed earlier (more
    # stable semantics), falling back to the higher-importance member.
    preferred_features: tuple[str, ...] = (
        "liquidity",
        "market_cap",
        "age_seconds",
        "price",
        "volume",
        "avg_trade",
        "unique_wallets",
    )


def _select_core(
    feature_names: list[str],
    X: np.ndarray,
    importance: list[FeatureImportance],
    config: ThresholdConfig,
) -> SelectionResult:
    """Rank -> prune -> cap, shared by both the in-memory and frame paths.

    1. correlation prune: of any pair with |r| above the threshold, keep the
       preferred (semantically stable) member, else the higher-ranked one
    2. low-variance prune
    3. cumulative walk in importance order until the cumulative target OR the
       hard max_features ceiling, never below min_features
    """
    if not importance:
        return SelectionResult(features=[], importance=[])
    if len(X) < config.min_features:
        return SelectionResult(
            features=[f.name for f in importance], importance=importance
        )

    col_idx = {name: i for i, name in enumerate(feature_names)}
    corr = _safe_corrcoef(X)
    preferred = {
        name: rank for rank, name in enumerate(config.preferred_features)
    }

    removed: set[int] = set()
    for i in range(len(importance)):
        if i in removed:
            continue
        ci = col_idx[importance[i].name]
        for j in range(i + 1, len(importance)):
            if j in removed:
                continue
            cj = col_idx[importance[j].name]
            if abs(corr[ci, cj]) <= config.correlation_threshold:
                continue
            # Correlated pair: keep the preferred (semantically stable) member,
            # or the higher-ranked one when neither is preferred.
            ri = preferred.get(importance[i].name, len(preferred))
            rj = preferred.get(importance[j].name, len(preferred))
            if rj < ri:
                removed.add(i)
                break
            removed.add(j)

    pruned = [f for idx, f in enumerate(importance) if idx not in removed]

    if pruned:
        pruned_names = [f.name for f in pruned]
        var_X = X[:, [col_idx[n] for n in pruned_names]]
        variances = np.var(var_X, axis=0)
        pruned = [
            f
            for idx, f in enumerate(pruned)
            if variances[idx] >= config.variance_threshold
        ]

    if len(pruned) < config.min_features:
        pruned = [
            f for idx, f in enumerate(importance) if idx not in removed
        ][: config.min_features]
        if len(pruned) < config.min_features:
            pruned = importance[: config.min_features]

    selected: list[str] = []
    cumulative = 0.0
    cap_reason = "all"
    for feat in pruned:
        selected.append(feat.name)
        cumulative += feat.importance
        if cumulative >= config.cumulative_importance:
            cap_reason = "cumulative"
            break
        if len(selected) >= config.max_features:
            cap_reason = "max_features"
            break

    if len(selected) < config.min_features:
        selected = [f.name for f in pruned[: config.min_features]]
        cap_reason = "min_features"

    return SelectionResult(features=selected, importance=importance, cap_reason=cap_reason)


class RandomForestSelector:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples: int = 100,
        cumulative_threshold: float = 0.70,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.cumulative_threshold = cumulative_threshold

    def select(self, dataset: TrainingDataset) -> SelectionResult:
        if len(dataset) < self.min_samples:
            return SelectionResult(features=[], importance=[])

        X, y, feature_names = _dataset_arrays(dataset)
        return self.select_from_arrays(X, y, feature_names)

    def select_from_arrays(
        self, X, y, feature_names: list[str]
    ) -> SelectionResult:
        if len(X) < self.min_samples:
            return SelectionResult(features=[], importance=[])

        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X, y)

        importance = [
            FeatureImportance(name=name, importance=imp)
            for name, imp in sorted(
                zip(feature_names, clf.feature_importances_, strict=False),
                key=lambda x: x[1],
                reverse=True,
            )
        ]

        return _select_core(
            feature_names,
            np.asarray(X, dtype=np.float64),
            importance,
            ThresholdConfig(cumulative_importance=self.cumulative_threshold),
        )


class CorrelationFilter:
    def __init__(self, config: ThresholdConfig | None = None):
        self.config = config or ThresholdConfig()

    def filter(
        self, dataset: TrainingDataset, importance: list[FeatureImportance]
    ) -> list[str]:
        if len(dataset) < self.config.min_features:
            return [f.name for f in importance]

        feature_names = [f.name for f in importance]
        X, _, _ = _dataset_arrays(dataset, feature_names)

        corr = _safe_corrcoef(X)
        removed: set[int] = set()

        for i in range(len(feature_names)):
            if i in removed:
                continue
            for j in range(i + 1, len(feature_names)):
                if j in removed:
                    continue
                if abs(corr[i, j]) > self.config.correlation_threshold:
                    removed.add(j)

        return [f for idx, f in enumerate(feature_names) if idx not in removed]


class LowVarianceFilter:
    def __init__(self, config: ThresholdConfig | None = None):
        self.config = config or ThresholdConfig()

    def filter(self, dataset: TrainingDataset, candidates: list[str]) -> list[str]:
        if len(dataset) < self.config.min_features:
            return candidates

        X, _, _ = _dataset_arrays(dataset, candidates)
        variances = np.var(X, axis=0)
        return [f for idx, f in enumerate(candidates) if variances[idx] >= self.config.variance_threshold]


class FeatureSelector:
    def __init__(
        self,
        rf_selector: RandomForestSelector | None = None,
        corr_filter: CorrelationFilter | None = None,
        var_filter: LowVarianceFilter | None = None,
        threshold_config: ThresholdConfig | None = None,
    ):
        tc = threshold_config or ThresholdConfig()
        self.rf_selector = rf_selector or RandomForestSelector(cumulative_threshold=tc.cumulative_importance)
        self.corr_filter = corr_filter or CorrelationFilter(tc)
        self.var_filter = var_filter or LowVarianceFilter(tc)

    def select(self, dataset: TrainingDataset, progress=None, task_id=None) -> SelectionResult:
        result = self.rf_selector.select(dataset)

        if progress is not None and task_id is not None:
            progress.update(task_id, advance=1)

        return result

    def select_from_frame(self, frame, label_column: str = "label", progress=None, task_id=None) -> SelectionResult:
        feature_names = [c for c in frame.columns if c != label_column]
        X = frame.select(feature_names).to_numpy()
        y = frame[label_column].to_numpy().astype(np.int32)

        result = self.rf_selector.select_from_arrays(X, y, feature_names)

        if progress is not None and task_id is not None:
            progress.update(task_id, advance=1)

        return result
