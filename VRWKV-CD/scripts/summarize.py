#!/usr/bin/env python3
"""Summarize completed seed JSON files as mean ± population std."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("oa", "kappa", "precision", "recall", "f1", "iou")


def selected_metrics(payload: dict) -> dict:
    protocol = payload["config"]["selection_protocol"]
    if protocol == "test_peak_1pct":
        evaluation = payload["test_peak"]["test_eval"]["raw"]
    else:
        evaluation = payload["protocol_selected"]["test_eval"]["raw"]
    return {
        "oa": evaluation["oa"],
        "kappa": evaluation["kappa"],
        "precision": evaluation.get("changed_precision", evaluation["precision"]),
        "recall": evaluation.get("changed_recall", evaluation["recall"]),
        "f1": evaluation.get("changed_f1", evaluation["f1"]),
        "iou": evaluation.get("changed_iou", evaluation["iou_change"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("summary.md"))
    args = parser.parse_args()
    rows = []
    for path in sorted(args.result_dir.rglob("seed_*.json")):
        try:
            payload = json.loads(path.read_text())
            metric = selected_metrics(payload)
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
        rows.append((path, metric))
    lines = ["# Experimental summary", "", f"Completed seeds: {len(rows)}", ""]
    if rows:
        values = np.asarray([[metric[key] for key in METRICS]
                             for _, metric in rows], dtype=np.float64) * 100
        lines += ["|Metric|Mean ± std (%)|", "|---|---:|"]
        for index, key in enumerate(METRICS):
            lines.append(
                f"|{key.upper()}|{values[:, index].mean():.4f} ± "
                f"{values[:, index].std(ddof=0):.4f}|")
        lines += ["", "## Per-seed results", "",
                  "|Result|" + "|".join(key.upper() for key in METRICS) + "|",
                  "|---|" + "|".join("---:" for _ in METRICS) + "|"]
        for path, metric in rows:
            cells = [100 * metric[key] for key in METRICS]
            lines.append(f"|{path}|" + "|".join(f"{x:.4f}" for x in cells) + "|")
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
