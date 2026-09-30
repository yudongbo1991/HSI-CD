"""Train-only low-capacity spectral physics calibration."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def spectral_features(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    delta = second - first
    absolute = np.abs(delta)
    eps = 1e-6
    cosine = (first * second).sum(1) / (
        np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1) + eps)
    return np.stack((
        1.0 - cosine, absolute.mean(1), absolute.max(1),
        np.sqrt(np.mean(delta * delta, axis=1)),
        absolute.mean(1) / (np.mean(np.abs(first) + np.abs(second), axis=1) + eps),
        delta.mean(1), delta.std(1)), axis=1)


def center_spectra(patches: np.ndarray) -> np.ndarray:
    """Return [sample, band] spectra from either spectra or flattened patches."""
    if patches.ndim == 2:
        return patches
    if patches.ndim != 3:
        raise ValueError(f"expected [N,C] or [N,C,P], got {patches.shape}")
    return patches[:, :, patches.shape[-1] // 2]


def grouped_spectral_features(first: np.ndarray, second: np.ndarray,
                              groups: int = 8) -> np.ndarray:
    """Compact change-shape descriptors over contiguous wavelength groups."""
    first, second = center_spectra(first), center_spectra(second)
    delta = second - first
    derivative = np.pad(np.abs(np.diff(delta, axis=1)), ((0, 0), (0, 1)))
    sources = (delta, np.abs(delta), delta * delta,
               np.abs(delta) / (np.abs(first) + np.abs(second) + 1e-4),
               derivative)
    parts = [spectral_features(first, second)]
    for source in sources:
        parts.append(np.stack([
            chunk.mean(1) for chunk in np.array_split(source, groups, axis=1)
        ], axis=1))
    return np.concatenate(parts, axis=1)


def grouped_spatial_features(first: np.ndarray, second: np.ndarray,
                             groups: int = 8) -> np.ndarray:
    """Grouped spectra plus low-capacity local evidence from the same patch.

    The spatial descriptors are groupwise means/stds rather than learned test
    adaptation, so fitting remains strictly confined to the 1% labelled set.
    """
    if first.ndim != 3 or second.ndim != 3:
        raise ValueError("grouped_spatial mode requires flattened patches")
    width = int(round(first.shape[-1] ** 0.5))
    if width * width != first.shape[-1]:
        raise ValueError("flattened patch area must be a square")
    delta = (second - first).reshape(first.shape[0], first.shape[1], width, width)
    absolute = np.abs(delta)
    radius = min(1, width // 2)
    center = width // 2
    local = absolute[:, :, center-radius:center+radius+1,
                     center-radius:center+radius+1]
    # Each statistic yields one value per contiguous spectral group.
    parts = [grouped_spectral_features(first, second, groups)]
    for source in (absolute.mean((2, 3)), absolute.std((2, 3)),
                   local.mean((2, 3)), local.std((2, 3)),
                   delta.mean((2, 3))):
        parts.append(np.stack([
            chunk.mean(1) for chunk in np.array_split(source, groups, axis=1)
        ], axis=1))
    return np.concatenate(parts, axis=1)


class TrainOnlyPhysicsCalibrator:
    def __init__(self, margin: float = 0.0, mode: str = "center"):
        self.margin = float(margin)
        if mode not in {"center", "grouped", "grouped_oof",
                        "grouped_spatial"}:
            raise ValueError("unsupported physics mode")
        self.mode = mode
        self.decision_threshold = 0.5
        self.logistic = make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=2000,
                                                  random_state=0))
        self.rbf = None
        self.blend = 0.5

    def fit(self, first_center: np.ndarray, second_center: np.ndarray,
            labels: np.ndarray):
        if self.mode == "grouped_spatial":
            features = grouped_spatial_features(first_center, second_center)
        elif self.mode in {"grouped", "grouped_oof"}:
            features = grouped_spectral_features(first_center, second_center)
        else:
            features = spectral_features(center_spectra(first_center),
                                         center_spectra(second_center))
        splitter = StratifiedKFold(5, shuffle=True, random_state=0)
        if self.mode in {"grouped", "grouped_oof", "grouped_spatial"}:
            search = GridSearchCV(
                make_pipeline(StandardScaler(), LogisticRegression(
                    max_iter=3000, random_state=0)),
                {"logisticregression__C": [0.001, 0.01, 0.1, 1.0, 10.0]},
                scoring="accuracy", n_jobs=5, cv=splitter, refit=True)
            search.fit(features, labels)
            self.grouped = search.best_estimator_
            if self.mode == "grouped_oof":
                oof = cross_val_predict(
                    clone(self.grouped), features, labels, cv=splitter,
                    method="predict_proba", n_jobs=5)[:, 1]
                candidates = np.linspace(0.25, 0.75, 201)
                scores = np.asarray([np.mean((oof >= threshold) == labels)
                                     for threshold in candidates])
                best = np.flatnonzero(scores == scores.max())
                chosen = best[np.argmin(np.abs(candidates[best] - 0.5))]
                self.decision_threshold = float(candidates[chosen])
            return {"samples": int(len(labels)), "mode": self.mode,
                    "cv": float(search.best_score_),
                    "params": search.best_params_, "margin": self.margin,
                    "decision_threshold": self.decision_threshold}
        search = GridSearchCV(
            make_pipeline(StandardScaler(), SVC(probability=True, random_state=0)),
            {"svc__C": [0.1, 1.0, 10.0, 100.0],
             "svc__gamma": ["scale", 0.1, 0.5, 1.0]},
            scoring="accuracy", n_jobs=8, cv=splitter, refit=True)
        self.logistic.fit(features, labels)
        search.fit(features, labels)
        self.rbf = search.best_estimator_
        return {"samples": int(len(labels)), "mode": self.mode,
                "rbf_cv": float(search.best_score_),
                "rbf_params": search.best_params_, "margin": self.margin}

    def probability(self, first_center: np.ndarray,
                    second_center: np.ndarray) -> np.ndarray:
        if self.mode == "grouped_spatial":
            # A full Hermiston test tensor is roughly 7.4 GB before temporary
            # spatial-difference arrays.  Extract descriptors in bounded
            # batches; this is mathematically identical to one-shot inference
            # while avoiding several simultaneous full-scene temporaries.
            chunks = []
            for start in range(0, len(first_center), 512):
                stop = min(start + 512, len(first_center))
                features = grouped_spatial_features(
                    first_center[start:stop], second_center[start:stop])
                chunks.append(self.grouped.predict_proba(features)[:, 1])
            return np.concatenate(chunks)
        if self.mode in {"grouped", "grouped_oof"}:
            return self.grouped.predict_proba(grouped_spectral_features(
                first_center, second_center))[:, 1]
        features = spectral_features(center_spectra(first_center),
                                     center_spectra(second_center))
        logistic = self.logistic.predict_proba(features)[:, 1]
        rbf = self.rbf.predict_proba(features)[:, 1]
        return (1.0 - self.blend) * logistic + self.blend * rbf

    def route(self, network_probability: np.ndarray,
              first_center: np.ndarray, second_center: np.ndarray,
              unreliable: np.ndarray = None):
        physics = self.probability(first_center, second_center)
        network_class = network_probability >= 0.5
        physics_class = physics >= self.decision_threshold
        if self.mode == "grouped_oof":
            threshold = self.decision_threshold
            physics_confidence = np.where(
                physics_class, (physics - threshold) / max(1.0 - threshold, 1e-6),
                (threshold - physics) / max(threshold, 1e-6))
            network_confidence = 2.0 * np.abs(network_probability - 0.5)
        else:
            physics_confidence = 2.0 * np.abs(physics - 0.5)
            network_confidence = 2.0 * np.abs(network_probability - 0.5)
        # Forward/reverse class disagreement is label-free epistemic evidence.
        # The two passes are already required by symmetric CD inference, so the
        # reliability rule adds neither a model branch nor another forward.
        if unreliable is not None:
            unreliable = np.asarray(unreliable, dtype=bool)
            if unreliable.shape != network_probability.shape:
                raise ValueError("unreliable mask must match probabilities")
            network_confidence = np.where(unreliable, 0.0,
                                          network_confidence)
        take = ((network_class != physics_class)
                & (physics_confidence > network_confidence + 2.0 * self.margin))
        return np.where(take, physics, network_probability), int(take.sum())
