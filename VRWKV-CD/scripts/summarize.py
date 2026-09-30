#!/usr/bin/env python3
"""Summarize final dual-branch results over completed seeds."""
import argparse
import json
from pathlib import Path
import numpy as np

KEYS = ("oa", "kappa", "precision", "recall", "f1", "iou")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("summary.md"))
    args = parser.parse_args()
    values = {"loose99": [], "strict7": []}
    for path in sorted(args.result_dir.glob("seed_*/result.json")):
        try:
            result = json.loads(path.read_text())
            for scope in values:
                values[scope].append([result[scope][key] for key in KEYS])
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    lines = ["# Final dual-branch results", "",
             "|Test scope|Seeds|OA|Kappa|Precision|Recall|F1|IoU|",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for scope, rows in values.items():
        if not rows:
            lines.append(f"|{scope}|0|—|—|—|—|—|—|")
            continue
        array = 100 * np.asarray(rows)
        cells = [f"{array[:, i].mean():.4f}±{array[:, i].std():.4f}"
                 for i in range(len(KEYS))]
        lines.append(f"|{scope}|{len(rows)}|" + "|".join(cells) + "|")
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
