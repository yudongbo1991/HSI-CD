"""Train the final dual-branch model once and evaluate two test scopes."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from core.model import VRWKVChangeDetector
from data import DATASETS, PairPatchDataset, load_split
from metrics import binary_metrics


DATASET_CONFIG = {
    "river": {"patch": 7, "spectral_weight": 0.5, "share_projection": True},
    "yancheng": {"patch": 7, "spectral_weight": 0.6, "share_projection": False},
    "bayarea": {"patch": 7, "spectral_weight": 0.4, "share_projection": False},
    "hermiston390": {"patch": 5, "spectral_weight": 0.5, "share_projection": False},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--patch", type=int, default=None,
                        help="patch size; default is dataset-specific")
    parser.add_argument("--spectral-weight", type=float, default=None,
                        help="spectral probability weight; spatial weight is 1-w")
    projection = parser.add_mutually_exclusive_group()
    projection.add_argument(
        "--share-spatial-projection", dest="share_spatial_projection",
        action="store_true", help="share the T1/T2 spatial input projection")
    projection.add_argument(
        "--no-share-spatial-projection", dest="share_spatial_projection",
        action="store_false", help="use independent T1/T2 input projections")
    parser.set_defaults(share_spatial_projection=None)
    parser.add_argument("--strict-radius", type=int, default=6)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(scene, positions, labels, patch, batch_size, shuffle, seed):
    return DataLoader(
        PairPatchDataset(scene.first, scene.second, positions, labels, patch),
        batch_size=batch_size, shuffle=shuffle, num_workers=0,
        pin_memory=True)


@torch.no_grad()
def evaluate(model, loader, device, return_prediction=False):
    model.eval()
    targets, predictions = [], []
    loss_sum = 0.0
    for first, second, target in loader:
        target_device = target.to(device, non_blocking=True)
        output = model(first.to(device, non_blocking=True),
                       second.to(device, non_blocking=True))["logits"]
        loss_sum += float(F.cross_entropy(
            output, target_device, reduction="sum"))
        targets.append(target.numpy())
        predictions.append(output.argmax(1).cpu().numpy())
    target = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    result = binary_metrics(target, prediction)
    result["loss"] = loss_sum / len(target)
    return (result, target, prediction) if return_prediction else result


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for first, second, target in loader:
        first = first.to(device, non_blocking=True)
        second = second.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(first, second)
        fused_loss = F.cross_entropy(output["logits"], target)
        spatial_loss = F.cross_entropy(output["spatial_logits"], target)
        spectral_loss = F.cross_entropy(output["spectral_logits"], target)
        loss = fused_loss + spatial_loss + spectral_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.detach()) * len(target)
        correct += int((output["logits"].argmax(1) == target).sum())
        total += len(target)
    return total_loss / total, correct / total


def save_error_map(path, shape, indices, target, prediction):
    canvas = np.full((shape[0] * shape[1], 3), 150, dtype=np.uint8)
    indices = np.ravel_multi_index(
        (indices[:, 0], indices[:, 1]), shape)
    canvas[indices[(target == 1) & (prediction == 1)]] = (0, 0, 0)
    canvas[indices[(target == 0) & (prediction == 0)]] = (255, 255, 255)
    canvas[indices[(target == 0) & (prediction == 1)]] = (230, 30, 30)
    canvas[indices[(target == 1) & (prediction == 0)]] = (30, 90, 230)
    Image.fromarray(canvas.reshape(*shape, 3)).save(path)


def main():
    args = parse_args()
    setting = DATASET_CONFIG[args.dataset]
    args.patch = setting["patch"] if args.patch is None else args.patch
    spectral_weight = (setting["spectral_weight"] if args.spectral_weight is None
                       else args.spectral_weight)
    share_projection = (setting["share_projection"]
                        if args.share_spatial_projection is None
                        else args.share_spatial_projection)
    if args.patch < 1 or args.patch % 2 == 0:
        raise ValueError("patch size must be a positive odd integer")
    if not 0.0 <= spectral_weight <= 1.0:
        raise ValueError("spectral weight must be in [0, 1]")
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    started = time.time()

    scene = load_split(args.dataset, args.seed, args.patch, args.strict_radius)
    train_loader = make_loader(
        scene, scene.train_positions, scene.train_labels, args.patch,
        args.batch_size, True, args.seed)
    validation_loader = make_loader(
        scene, scene.validation_positions, scene.validation_labels, args.patch,
        args.test_batch_size, False, args.seed)
    loose_loader = make_loader(
        scene, scene.loose_test_positions, scene.loose_test_labels, args.patch,
        args.test_batch_size, False, args.seed)
    strict_loader = make_loader(
        scene, scene.strict_test_positions, scene.strict_test_labels, args.patch,
        args.test_batch_size, False, args.seed)

    model = VRWKVChangeDetector(
        bands=scene.first.shape[-1], patch_size=args.patch,
        spectral_weight=spectral_weight,
        share_spatial_projection=share_projection).to(device)
    decay, regular = [], []
    for name, parameter in model.named_parameters():
        (decay if "spatial_decay" in name else regular).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": regular, "weight_decay": args.weight_decay},
        {"params": decay, "weight_decay": 0.0},
    ], lr=args.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.lr * 0.01)

    best = {"key": (-1.0, float("-inf")), "epoch": -1, "state": None}
    history = []
    log_path = args.output / "train.log"
    log_path.write_text("")
    for epoch in range(1, args.epochs + 1):
        loss, train_oa = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        row = {"epoch": epoch, "loss": loss, "train_oa": train_oa}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            validation = evaluate(model, validation_loader, device)
            row["validation"] = validation
            selection_key = (validation["oa"], -validation["loss"])
            if selection_key > best["key"]:
                best = {"key": selection_key, "epoch": epoch,
                        "state": copy.deepcopy(model.state_dict())}
            message = (f"Epoch {epoch:03d} | loss {loss:.6f} | "
                       f"train OA {train_oa:.6f} | "
                       f"validation OA {validation['oa']:.6f}")
        else:
            message = (f"Epoch {epoch:03d} | loss {loss:.6f} | "
                       f"train OA {train_oa:.6f}")
        print(message, flush=True)
        with log_path.open("a") as stream:
            stream.write(message + "\n")
        history.append(row)

    model.load_state_dict(best["state"])
    loose, loose_target, loose_prediction = evaluate(
        model, loose_loader, device, True)
    strict, strict_target, strict_prediction = evaluate(
        model, strict_loader, device, True)
    save_error_map(args.output / "loose99_error_map.png",
                   scene.binary_labels.shape, scene.loose_test_positions,
                   loose_target, loose_prediction)
    save_error_map(args.output / "strict7_error_map.png",
                   scene.binary_labels.shape, scene.strict_test_positions,
                   strict_target, strict_prediction)
    np.savez_compressed(
        args.output / "predictions.npz",
        train_positions=scene.train_positions,
        validation_positions=scene.validation_positions,
        loose_test_positions=scene.loose_test_positions,
        loose_test_labels=loose_target,
        loose_prediction=loose_prediction,
        strict_test_positions=scene.strict_test_positions,
        strict_test_labels=strict_target,
        strict_prediction=strict_prediction)
    result = {
        "dataset": args.dataset, "seed": args.seed,
        "selection": "best validation OA",
        "selected_epoch": best["epoch"],
        "loose99": loose, "strict7": strict,
        "config": {
            "patch": args.patch, "strict_radius": args.strict_radius,
            "train_fraction": 0.005, "validation_fraction": 0.005,
            "normalization": "training patches only",
            "epochs": args.epochs, "eval_every": args.eval_every,
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
            "learning_rate": args.lr, "weight_decay": args.weight_decay,
            "spectral_weight": spectral_weight,
            "spatial_weight": 1.0 - spectral_weight,
            "share_spatial_projection": share_projection,
        },
        "counts": {
            "train": len(scene.train_labels),
            "validation": len(scene.validation_labels),
            "loose99": len(scene.loose_test_labels),
            "strict7": len(scene.strict_test_labels),
        },
        "history": history,
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"selected_epoch": best["epoch"],
                      "loose99": loose, "strict7": strict}, indent=2))


if __name__ == "__main__":
    main()
