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


class RandomForestSelector:
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, min_samples: int = 100, cumulative_threshold: float = 0.95):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.cumulative_threshold = cumulative_threshold

    def select(self, dataset: TrainingDataset) -> SelectionResult:
        if len(dataset) < self.min_samples:
            return SelectionResult(features=[], importance=[])

        from sklearn.ensemble import RandomForestClassifier

        X, y, feature_names = _dataset_arrays(dataset)

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

        cumulative = 0.0
        selected: list[str] = []
        for feat in importance:
            cumulative += feat.importance
            selected.append(feat.name)
            if cumulative >= self.cumulative_threshold:
                break

        if not selected:
            selected = [f.name for f in importance]

        return SelectionResult(features=selected, importance=importance)


@dataclass(slots=True)
class ThresholdConfig:
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01
    min_features: int = 5
    cumulative_importance: float = 0.95


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
        rf_result = self.rf_selector.select(dataset)
        if not rf_result.features:
            return rf_result

        if progress is not None and task_id is not None:
            progress.update(task_id, description="Filtering correlated features...")

        filtered = self.corr_filter.filter(dataset, rf_result.importance)
        filtered = self.var_filter.filter(dataset, filtered)

        if not filtered:
            filtered = rf_result.features[:5]

        filtered_importance = [f for f in rf_result.importance if f.name in filtered]

        if progress is not None and task_id is not None:
            progress.update(task_id, advance=1)

        return SelectionResult(features=filtered, importance=filtered_importance)

    def select_from_frame(self, frame, label_column: str = "label", progress=None, task_id=None) -> SelectionResult:
        import polars as pl

        feature_names = [c for c in frame.columns if c != label_column]
        X = frame.select(feature_names).to_numpy()
        y = frame[label_column].to_numpy().astype(np.int32)

        if len(X) < self.rf_selector.min_samples:
            return SelectionResult(features=[], importance=[])

        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(
            n_estimators=self.rf_selector.n_estimators,
            max_depth=self.rf_selector.max_depth,
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

        cumulative = 0.0
        selected: list[str] = []
        for feat in importance:
            cumulative += feat.importance
            selected.append(feat.name)
            if cumulative >= self.rf_selector.cumulative_threshold:
                break
        if not selected:
            selected = [f.name for f in importance]

        if progress is not None and task_id is not None:
            progress.update(task_id, description="Filtering correlated features...")

        corr_X = frame.select([f.name for f in importance]).to_numpy()
        corr = _safe_corrcoef(corr_X)
        removed: set[int] = set()
        n = len(importance)
        for i in range(n):
            if i in removed:
                continue
            for j in range(i + 1, n):
                if j in removed:
                    continue
                if abs(corr[i, j]) > self.corr_filter.config.correlation_threshold:
                    removed.add(j)

        filtered = [f.name for idx, f in enumerate(importance) if idx not in removed]

        var_X = frame.select(filtered).to_numpy()
        variances = np.var(var_X, axis=0)
        filtered = [f for idx, f in enumerate(filtered) if variances[idx] >= self.var_filter.config.variance_threshold]

        if not filtered:
            filtered = selected[:5]

        filtered_importance = [f for f in importance if f.name in filtered]

        if progress is not None and task_id is not None:
            progress.update(task_id, advance=1)

        return SelectionResult(features=filtered, importance=filtered_importance)
