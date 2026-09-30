"""Four-dataset loading, strict 1% split, and lazy paired patches."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from scipy.ndimage import binary_dilation
from torch.utils.data import Dataset


DATA_ROOT = Path(os.environ.get("HSICD_DATA_ROOT", "data")).expanduser().resolve()
DATASETS = ("river", "yancheng", "bayarea", "hermiston390")


@dataclass(frozen=True)
class SceneSplit:
    first: np.ndarray
    second: np.ndarray
    binary_labels: np.ndarray
    train_positions: np.ndarray
    train_labels: np.ndarray
    validation_positions: np.ndarray
    validation_labels: np.ndarray
    loose_test_positions: np.ndarray
    loose_test_labels: np.ndarray
    strict_test_positions: np.ndarray
    strict_test_labels: np.ndarray
    strict_keep_mask: np.ndarray


def _load_scene(dataset: str):
    if dataset == "river":
        root = DATA_ROOT / "River"
        first = loadmat(root / "river_before.mat")["river_before"]
        second = loadmat(root / "river_after.mat")["river_after"]
        labels = loadmat(root / "groundtruth.mat")["lakelabel_v1"]
        return first, second, labels, False
    if dataset == "yancheng":
        root = DATA_ROOT / "farmland2"
        first = loadmat(root / "farm06.mat")["imgh"]
        second = loadmat(root / "farm07.mat")["imghl"]
        labels = loadmat(root / "label.mat")["label"]
        return first, second, labels, False
    if dataset == "bayarea":
        root = DATA_ROOT / "BayArea"
        first = loadmat(root / "Bay_Area_2013.mat")["HypeRvieW"]
        second = loadmat(root / "Bay_Area_2015.mat")["HypeRvieW"]
        labels = loadmat(root / "bayArea_gtChanges2.mat")["HypeRvieW"]
        return first, second, labels, True
    if dataset == "hermiston390":
        root = DATA_ROOT / "Hermiston390_CITIUS/Hermiston"
        first = loadmat(root / "hermiston2004.mat")["HypeRvieW"]
        second = loadmat(root / "hermiston2007.mat")["HypeRvieW"]
        labels = loadmat(root / "rdChangesHermiston_5classes.mat")[
            "gt5clasesHermiston"]
        return first, second, (labels > 0).astype(np.uint8), False
    raise ValueError(f"unknown dataset {dataset!r}; choose from {DATASETS}")


def _classwise_budget(raw_labels, ignored, seed, fraction=0.01):
    if ignored:
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
    # Keep the historical RNG order for exact reproducibility.
    unchanged_pick = rng.choice(
        len(unchanged), int(fraction * len(unchanged)), replace=False)
    changed_pick = rng.choice(
        len(changed), int(fraction * len(changed)), replace=False)
    budget_positions = np.concatenate(
        (changed[changed_pick], unchanged[unchanged_pick]))
    budget_labels = np.concatenate((
        np.zeros(len(changed_pick), dtype=np.int64),
        np.ones(len(unchanged_pick), dtype=np.int64)))
    selected = np.zeros(raw_labels.shape, dtype=bool)
    selected[budget_positions[:, 0], budget_positions[:, 1]] = True
    loose_positions = np.concatenate((
        changed[~selected[changed[:, 0], changed[:, 1]]],
        unchanged[~selected[unchanged[:, 0], unchanged[:, 1]]]))
    loose_labels = np.concatenate((
        np.zeros(np.sum(~selected[changed[:, 0], changed[:, 1]]), dtype=np.int64),
        np.ones(np.sum(~selected[unchanged[:, 0], unchanged[:, 1]]), dtype=np.int64)))
    return binary, budget_positions, budget_labels, loose_positions, loose_labels


def _split_budget(positions, labels, seed):
    rng = np.random.RandomState(seed + 104729)
    train_indices, validation_indices = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        indices = indices[rng.permutation(len(indices))]
        count = min(max(1, int(round(0.5 * len(indices)))), len(indices) - 1)
        validation_indices.extend(indices[:count])
        train_indices.extend(indices[count:])
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    return (positions[train_indices], labels[train_indices],
            positions[validation_indices], labels[validation_indices])


def reflect_pad(cube, patch):
    radius = patch // 2
    return np.pad(cube, ((radius, radius), (radius, radius), (0, 0)),
                  mode="symmetric").astype(np.float32, copy=False)


def _normalize_from_training_patches(first, second, positions, patch):
    first_pad, second_pad = reflect_pad(first, patch), reflect_pad(second, patch)
    low = np.full(first.shape[2], np.inf, dtype=np.float64)
    high = np.full(first.shape[2], -np.inf, dtype=np.float64)
    for row, column in positions:
        patch1 = first_pad[row:row + patch, column:column + patch]
        patch2 = second_pad[row:row + patch, column:column + patch]
        low = np.minimum(low, np.minimum(patch1.min((0, 1)), patch2.min((0, 1))))
        high = np.maximum(high, np.maximum(patch1.max((0, 1)), patch2.max((0, 1))))
    span = np.maximum(high - low, 1e-12)
    return (((first - low) / span).astype(np.float32),
            ((second - low) / span).astype(np.float32),
            low, high)


def load_split(dataset, seed, patch=7, strict_radius=6):
    first, second, raw_labels, ignored = _load_scene(dataset)
    binary, budget_pos, budget_y, loose_pos, loose_y = _classwise_budget(
        raw_labels, ignored, seed)
    train_pos, train_y, val_pos, val_y = _split_budget(
        budget_pos, budget_y, seed)
    first, second, _, _ = _normalize_from_training_patches(
        first, second, train_pos, patch)
    supervised = np.zeros(raw_labels.shape, dtype=bool)
    supervised[budget_pos[:, 0], budget_pos[:, 1]] = True
    forbidden = binary_dilation(
        supervised, structure=np.ones((2 * strict_radius + 1,) * 2, dtype=bool))
    keep = ~forbidden[loose_pos[:, 0], loose_pos[:, 1]]
    return SceneSplit(
        first, second, binary, train_pos, train_y, val_pos, val_y,
        loose_pos, loose_y, loose_pos[keep], loose_y[keep], keep)


class PairPatchDataset(Dataset):
    def __init__(self, first, second, positions, labels, patch=7):
        self.first = reflect_pad(first, patch)
        self.second = reflect_pad(second, patch)
        self.positions = np.asarray(positions, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.patch = patch

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        row, column = self.positions[index]
        first = self.first[row:row + self.patch, column:column + self.patch]
        second = self.second[row:row + self.patch, column:column + self.patch]
        first = np.ascontiguousarray(first.transpose(2, 0, 1).reshape(
            first.shape[2], -1))
        second = np.ascontiguousarray(second.transpose(2, 0, 1).reshape(
            second.shape[2], -1))
        return (torch.from_numpy(first), torch.from_numpy(second),
                torch.tensor(self.labels[index], dtype=torch.long))
