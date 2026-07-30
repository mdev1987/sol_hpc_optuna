from __future__ import annotations

from dataclasses import dataclass, field

from dataset_builder import TrainingDataset


@dataclass(slots=True)
class FeatureImportance:
    name: str
    importance: float


@dataclass(slots=True)
class SelectionResult:
    features: list[str]
    importance: list[FeatureImportance]


class RandomForestSelector:
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, min_samples: int = 100, cumulative_threshold: float = 0.95):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.cumulative_threshold = cumulative_threshold

    def select(self, dataset: TrainingDataset) -> SelectionResult:
        if len(dataset) < self.min_samples:
            return SelectionResult(features=[], importance=[])

        import numpy as np
        from sklearn.ensemble import RandomForestClassifier

        feature_names = list(dataset.samples[0].features.keys())
        X = np.array([[s.features[f] for f in feature_names] for s in dataset.samples], dtype=np.float64)
        y = np.array([s.label for s in dataset.samples], dtype=np.int32)

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

        import numpy as np

        feature_names = [f.name for f in importance]
        X = np.array(
            [[s.features[f] for f in feature_names] for s in dataset.samples],
            dtype=np.float64,
        )

        corr = np.corrcoef(X.T)
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

        import numpy as np

        X = np.array(
            [[s.features[f] for f in candidates] for s in dataset.samples],
            dtype=np.float64,
        )
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
