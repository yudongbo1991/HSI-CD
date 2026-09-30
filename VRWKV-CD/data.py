"""Strict class-wise 1% splits and lazy paired-patch loading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


DATA_ROOT = Path(os.environ.get("HSICD_DATA_ROOT", "data")).expanduser().resolve()
DEFAULT_DATA = DATA_ROOT / "Hermiston/USA_Change_Dataset.mat"
DATASET_NAMES = ("hermiston", "hermiston390", "farmland", "yancheng",
                 "river", "bayarea", "barbara")


@dataclass(frozen=True)
class HermistonSplit:
    first: np.ndarray
    second: np.ndarray
    labels: np.ndarray
    train_positions: np.ndarray
    train_labels: np.ndarray
    test_positions: np.ndarray
    test_labels: np.ndarray


def _joint_band_normalize(first: np.ndarray, second: np.ndarray):
    low = np.minimum(first.min((0, 1)), second.min((0, 1)))
    high = np.maximum(first.max((0, 1)), second.max((0, 1)))
    span = np.maximum(high - low, 1e-12)
    return (((first - low) / span).astype(np.float32),
            ((second - low) / span).astype(np.float32), low, high)


def _training_patch_band_normalize(first: np.ndarray, second: np.ndarray,
                                   train_positions: np.ndarray, patch: int):
    """Fit joint T1/T2 per-band min/max on training patches only."""
    if patch < 1 or patch % 2 != 1:
        raise ValueError(f"normalization patch must be positive and odd, got {patch}")
    padded_first = reflect_pad(first, patch)
    padded_second = reflect_pad(second, patch)
    low = np.full(first.shape[2], np.inf, dtype=np.float64)
    high = np.full(first.shape[2], -np.inf, dtype=np.float64)
    for row, column in np.asarray(train_positions, dtype=np.int64):
        p1 = padded_first[row:row + patch, column:column + patch]
        p2 = padded_second[row:row + patch, column:column + patch]
        low = np.minimum(low, np.minimum(p1.min((0, 1)), p2.min((0, 1))))
        high = np.maximum(high, np.maximum(p1.max((0, 1)), p2.max((0, 1))))
    span = np.maximum(high - low, 1e-12)
    # Do not clip unseen values: preserve distribution shift outside train range.
    return (((first - low) / span).astype(np.float32),
            ((second - low) / span).astype(np.float32), low, high)


def _normalize_pair(first, second, train_positions, mode, patch):
    if mode == "scene_joint":
        return _joint_band_normalize(first, second)
    if mode == "train_patches":
        return _training_patch_band_normalize(first, second, train_positions, patch)
    raise ValueError(f"unknown input normalization mode {mode!r}")


def _normalization_fit_positions(train_positions: np.ndarray,
                                 train_labels: np.ndarray,
                                 protocol: str, seed: int) -> np.ndarray:
    """Return only the samples actually optimized by the selected protocol.

    This mirrors ``train.split_labelled_pool`` exactly.  In particular, held-
    out validation centers must not contribute pixels to normalization.
    """
    if protocol == "test_peak_1pct":
        return np.asarray(train_positions, dtype=np.int64)
    validation_fraction = {
        "val10_of_1pct": 0.10,
        "val50_of_1pct": 0.50,
    }[protocol]
    rng = np.random.RandomState(seed + 104729)
    fit_indices = []
    for label in np.unique(train_labels):
        indices = np.flatnonzero(train_labels == label)
        indices = indices[rng.permutation(len(indices))]
        validation_count = max(1, int(round(validation_fraction * len(indices))))
        validation_count = min(validation_count, len(indices) - 1)
        fit_indices.extend(indices[validation_count:])
    return np.asarray(train_positions, dtype=np.int64)[
        np.asarray(fit_indices, dtype=np.int64)]


def load_hermiston(seed: int, fraction: float = 0.01,
                   path: Path = DEFAULT_DATA,
                   split_output: Path | None = None,
                   normalization: str = "scene_joint",
                   normalization_patch: int = 11) -> HermistonSplit:
    """Reproduce the original class-wise NumPy split exactly.

    Class 0 (unchanged) becomes label index 1 and class 1 (changed) becomes
    label index 0, matching the historical training and confusion counts.
    Every labelled pixel not selected for training is used for testing.
    """
    raw = loadmat(str(path))
    first, second = raw["T1"], raw["T2"]
    binary = raw["Binary"].copy()
    unchanged = np.argwhere(binary == 0)
    changed = np.argwhere(binary == 1)
    rng = np.random.RandomState(seed)
    unchanged_pick = rng.choice(len(unchanged), int(fraction * len(unchanged)),
                                replace=False)
    changed_pick = rng.choice(len(changed), int(fraction * len(changed)),
                              replace=False)
    train_unchanged = unchanged[unchanged_pick]
    train_changed = changed[changed_pick]
    train_positions = np.concatenate((train_changed, train_unchanged), axis=0)
    train_labels = np.concatenate((
        np.zeros(len(train_changed), dtype=np.int64),
        np.ones(len(train_unchanged), dtype=np.int64)))
    selected = np.zeros(binary.shape, dtype=bool)
    selected[train_positions[:, 0], train_positions[:, 1]] = True
    test_changed = np.argwhere((binary == 1) & ~selected)
    test_unchanged = np.argwhere((binary == 0) & ~selected)
    test_positions = np.concatenate((test_changed, test_unchanged), axis=0)
    test_labels = np.concatenate((
        np.zeros(len(test_changed), dtype=np.int64),
        np.ones(len(test_unchanged), dtype=np.int64)))
    if len(train_positions) != 739 or len(test_positions) != 73248:
        raise RuntimeError(
            f"strict split violated: train={len(train_positions)}, test={len(test_positions)}")
    first, second, norm_low, norm_high = _normalize_pair(
        first, second, train_positions, normalization, normalization_patch)
    if split_output is not None:
        split_output = Path(split_output)
        split_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(split_output, seed=np.int64(seed),
                            train_positions=train_positions,
                            train_labels=train_labels,
                            test_positions=test_positions,
                            test_labels=test_labels,
                            normalization=normalization,
                            normalization_patch=np.int64(normalization_patch),
                            normalization_low=norm_low,
                            normalization_high=norm_high)
    return HermistonSplit(first, second, binary, train_positions, train_labels,
                          test_positions, test_labels)


def _load_named_pair(dataset: str):
    """Load the exact files used by the existing comparison implementations."""
    base = DATA_ROOT
    if dataset == "hermiston":
        raw = loadmat(str(base / "Hermiston/USA_Change_Dataset.mat"))
        return raw["T1"], raw["T2"], raw["Binary"], False
    if dataset == "hermiston390":
        # CITIUS/USC Hermiston scene used by the 390x200x242 literature.
        # Preserve the released five-change-class GT on disk and derive the
        # binary task only in memory: class 0 is unchanged, classes 1--5 are
        # changed.  This is a different cube from the historical USA-307
        # benchmark above and must never silently replace it.
        hermiston = base / "Hermiston390_CITIUS/Hermiston"
        first = loadmat(str(hermiston / "hermiston2004.mat"))["HypeRvieW"]
        second = loadmat(str(hermiston / "hermiston2007.mat"))["HypeRvieW"]
        multiclass = loadmat(str(
            hermiston / "rdChangesHermiston_5classes.mat"))[
                "gt5clasesHermiston"]
        labels = (multiclass > 0).astype(np.uint8)
        if (first.shape != (390, 200, 242)
                or second.shape != first.shape
                or labels.shape != first.shape[:2]
                or int(labels.sum()) != 9986):
            raise RuntimeError(
                "CITIUS Hermiston-390 integrity check failed: "
                f"first={first.shape}, second={second.shape}, "
                f"labels={labels.shape}, changed={int(labels.sum())}")
        return first, second, labels, False
    if dataset == "farmland":
        # This is the 420x140x154 China/Farmland pair used by MFCEN and SFIEET,
        # rather than the alternate 450x140 Yancheng representation.
        raw = loadmat(str(base / "farmland/China_Change_Dataset.mat"))
        return raw["T1"], raw["T2"], raw["Binary"], False
    if dataset == "yancheng":
        # Distinct 450x140x155 Farmland/Yancheng version used by GTransCD,
        # AIWSEN, QSCDNet, WDP-Mamba, SFIEET and CGMNet.
        first = loadmat(str(base / "farmland2/farm06.mat"))["imgh"]
        second = loadmat(str(base / "farmland2/farm07.mat"))["imghl"]
        labels = loadmat(str(base / "farmland2/label.mat"))["label"]
        return first, second, labels, False
    if dataset == "river":
        river = base / "River"
        first = loadmat(str(river / "river_before.mat"))["river_before"]
        second = loadmat(str(river / "river_after.mat"))["river_after"]
        labels = loadmat(str(river / "groundtruth.mat"))["lakelabel_v1"]
        return first, second, labels, False
    if dataset == "bayarea":
        first = loadmat(str(base / "BayArea/Bay_Area_2013.mat"))["HypeRvieW"]
        second = loadmat(str(base / "BayArea/Bay_Area_2015.mat"))["HypeRvieW"]
        labels = loadmat(str(base / "BayArea/bayArea_gtChanges2.mat"))["HypeRvieW"]
        return first, second, labels, True
    if dataset == "barbara":
        first = loadmat(str(base / "Barbara/barbara_2013.mat"))["HypeRvieW"]
        second = loadmat(str(base / "Barbara/barbara_2014.mat"))["HypeRvieW"]
        labels = loadmat(str(base / "Barbara/barbara_gtChanges.mat"))["HypeRvieW"]
        return first, second, labels, True
    raise ValueError(f"unknown dataset {dataset!r}; choose {DATASET_NAMES}")


def load_benchmark(dataset: str, seed: int, fraction: float = 0.01,
                   split_output: Path | None = None,
                   normalization: str = "scene_joint",
                   normalization_patch: int = 11,
                   normalization_protocol: str = "test_peak_1pct") -> HermistonSplit:
    """Use exactly 1% of each class; test on every remaining labelled pixel.

    Binary datasets encode unchanged/change as 0/1 (River uses 0/255).
    BayArea and Barbara use 0 for ignored pixels and 1/2 for change/unchanged.
    Internally target 0 is changed and target 1 is unchanged, preserving the
    historical Hermiston metric convention.
    """
    dataset = dataset.lower()
    first, second, raw_labels, has_ignored = _load_named_pair(dataset)
    if first.shape != second.shape or first.shape[:2] != raw_labels.shape:
        raise RuntimeError(
            f"{dataset} shape mismatch: {first.shape}, {second.shape}, {raw_labels.shape}")
    if has_ignored:
        changed = np.argwhere(raw_labels == 1)
        unchanged = np.argwhere(raw_labels == 2)
        binary = np.full(raw_labels.shape, 255, dtype=np.uint8)
        binary[unchanged[:, 0], unchanged[:, 1]] = 0
        binary[changed[:, 0], changed[:, 1]] = 1
    else:
        changed_value = raw_labels.max()
        changed = np.argwhere(raw_labels == changed_value)
        unchanged = np.argwhere(raw_labels == 0)
        binary = (raw_labels == changed_value).astype(np.uint8)
    rng = np.random.RandomState(seed)
    # Preserve the historical Hermiston RNG consumption exactly: sample the
    # unchanged population first, then the changed population.
    unchanged_pick = rng.choice(len(unchanged), int(fraction * len(unchanged)), replace=False)
    changed_pick = rng.choice(len(changed), int(fraction * len(changed)), replace=False)
    train_changed, train_unchanged = changed[changed_pick], unchanged[unchanged_pick]
    train_positions = np.concatenate((train_changed, train_unchanged), axis=0)
    train_labels = np.concatenate((np.zeros(len(train_changed), dtype=np.int64),
                                   np.ones(len(train_unchanged), dtype=np.int64)))
    selected = np.zeros(raw_labels.shape, dtype=bool)
    selected[train_positions[:, 0], train_positions[:, 1]] = True
    test_changed = changed[~selected[changed[:, 0], changed[:, 1]]]
    test_unchanged = unchanged[~selected[unchanged[:, 0], unchanged[:, 1]]]
    test_positions = np.concatenate((test_changed, test_unchanged), axis=0)
    test_labels = np.concatenate((np.zeros(len(test_changed), dtype=np.int64),
                                  np.ones(len(test_unchanged), dtype=np.int64)))
    if len(train_changed) != int(fraction * len(changed)) or len(
            train_unchanged) != int(fraction * len(unchanged)):
        raise RuntimeError(f"{dataset} strict class-wise split violated")
    normalization_positions = _normalization_fit_positions(
        train_positions, train_labels, normalization_protocol, seed)
    first, second, norm_low, norm_high = _normalize_pair(
        first, second, normalization_positions, normalization,
        normalization_patch)
    if split_output is not None:
        split_output = Path(split_output)
        split_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(split_output, dataset=dataset, seed=np.int64(seed),
                            train_positions=train_positions,
                            train_labels=train_labels,
                            test_positions=test_positions,
                            test_labels=test_labels,
                            normalization=normalization,
                            normalization_protocol=normalization_protocol,
                            normalization_fit_positions=normalization_positions,
                            normalization_patch=np.int64(normalization_patch),
                            normalization_low=norm_low,
                            normalization_high=norm_high)
    return HermistonSplit(first, second, binary, train_positions, train_labels,
                          test_positions, test_labels)


def reflect_pad(cube: np.ndarray, patch: int) -> np.ndarray:
    radius = patch // 2
    # Historical mirror_hsi includes the edge pixel itself, which corresponds
    # to NumPy's symmetric (not reflect) convention.
    return np.pad(cube, ((radius, radius), (radius, radius), (0, 0)),
                  mode="symmetric").astype(np.float32, copy=False)


class LazyPairDataset(Dataset):
    def __init__(self, first: np.ndarray, second: np.ndarray,
                 positions: np.ndarray, labels: np.ndarray, patch: int,
                 already_padded: bool = False):
        self.first = first if already_padded else reflect_pad(first, patch)
        self.second = second if already_padded else reflect_pad(second, patch)
        self.positions = np.asarray(positions, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.patch = patch

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        row, column = self.positions[index]
        first = self.first[row:row + self.patch, column:column + self.patch]
        second = self.second[row:row + self.patch, column:column + self.patch]
        first = np.ascontiguousarray(first.transpose(2, 0, 1).reshape(
            first.shape[2], -1))
        second = np.ascontiguousarray(second.transpose(2, 0, 1).reshape(
            second.shape[2], -1))
        return (torch.from_numpy(first), torch.from_numpy(second),
                torch.tensor(self.labels[index], dtype=torch.long))
