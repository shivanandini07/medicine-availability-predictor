"""Pure NumPy Random Forest fallback when scikit-learn native extensions are unavailable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class _Node:
    feature_index: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None
    value: float | None = None


class NumpyStandardScaler:
    """Minimal scaler matching sklearn StandardScaler interface."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "NumpyStandardScaler":
        self.mean_ = x.mean(axis=0)
        std = x.std(axis=0)
        std[std == 0] = 1.0
        self.scale_ = std
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


class NumpyDecisionTree:
    """Depth-limited binary decision tree for classification."""

    def __init__(self, max_depth: int = 8, min_samples_leaf: int = 5, seed: int = 0) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.root: _Node | None = None
        self.rng = np.random.default_rng(seed)

    def _gini(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        p = y.mean()
        return 2 * p * (1 - p)

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
        n_features = x.shape[1]
        best_gini = float("inf")
        best_feature = 0
        best_threshold = 0.0
        best_score = 0.0

        for feature in range(n_features):
            values = x[:, feature]
            thresholds = np.unique(values)
            if len(thresholds) > 32:
                thresholds = np.quantile(values, np.linspace(0.05, 0.95, 16))

            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue

                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])
                weighted = (left_mask.sum() * left_gini + right_mask.sum() * right_gini) / len(y)

                if weighted < best_gini:
                    best_gini = weighted
                    best_feature = feature
                    best_threshold = float(threshold)
                    best_score = y.mean()

        return best_feature, best_threshold, best_score

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        node = _Node(value=float(y.mean()))
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_leaf * 2:
            return node

        feature, threshold, _ = self._best_split(x, y)
        left_mask = x[:, feature] <= threshold
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return node

        node.feature_index = feature
        node.threshold = threshold
        node.left = self._build(x[left_mask], y[left_mask], depth + 1)
        node.right = self._build(x[right_mask], y[right_mask], depth + 1)
        return node

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyDecisionTree":
        self.root = self._build(x, y, depth=0)
        return self

    def _predict_row(self, node: _Node, row: np.ndarray) -> float:
        if node.feature_index is None or node.left is None or node.right is None:
            return float(node.value or 0.0)
        if row[node.feature_index] <= node.threshold:
            return self._predict_row(node.left, row)
        return self._predict_row(node.right, row)

    def predict_proba_rows(self, x: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Tree has not been fitted.")
        probs = np.array([self._predict_row(self.root, row) for row in x])
        return np.column_stack([1 - probs, probs])


class NumpyRandomForestClassifier:
    """Bootstrap aggregated random forest using only NumPy."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees: list[NumpyDecisionTree] = []
        self.rng = np.random.default_rng(random_state)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyRandomForestClassifier":
        n_samples = x.shape[0]
        self.trees = []
        for i in range(self.n_estimators):
            indices = self.rng.integers(0, n_samples, n_samples)
            tree = NumpyDecisionTree(max_depth=self.max_depth, seed=self.random_state + i)
            tree.fit(x[indices], y[indices])
            self.trees.append(tree)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("Forest has not been fitted.")
        probas = np.mean([tree.predict_proba_rows(x) for tree in self.trees], axis=0)
        return probas

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


class NumpyModelPipeline:
    """Pipeline wrapper compatible with sklearn and joblib persistence."""

    def __init__(self, n_estimators: int = 150, max_depth: int = 12, random_state: int = 42) -> None:
        self.scaler = NumpyStandardScaler()
        self.classifier = NumpyRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        self.steps: list[tuple[str, Any]] = [
            ("scaler", self.scaler),
            ("classifier", self.classifier),
        ]

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyModelPipeline":
        x_scaled = self.scaler.fit_transform(x)
        self.classifier.fit(x_scaled, y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler.transform(x)
        return self.classifier.predict_proba(x_scaled)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classifier.predict(self.scaler.transform(x))
