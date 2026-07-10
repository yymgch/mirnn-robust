# %%
"""Aggregate training acceptance logs and select models for downstream analysis.

This implements the model-selection flow requested by Reviewer 1 (point 2):
for each training condition it reports how many models were trained, how many
passed the R^2 > threshold criterion, the R^2 distribution before and after
selection, and the final sample size, and it emits the list of accepted model
IDs that the noise- and ablation-robustness scripts consume.

Inputs
------
All ``training_acceptance_log_<condition>_<start-index>.csv`` files found in the
input directory (written by ``train_and_save_models.py``). Each row records one
training attempt with columns:
``attempt, model_id, condition, accepted, mean_r2, reason, epochs,
learning_rate, r2_threshold``.

Outputs
-------
- ``accepted_models.json``: ``{condition: [model_id, ...]}`` with the
  ``model_`` prefix stripped, ready for the figure scripts.
- ``model_selection_summary.csv``: the per-condition reporting table
  (total trained, accepted, acceptance rate, R^2 mean/sd before and after
  selection) for the manuscript / supplementary.
- ``model_selection_r2.npz``: per-condition R^2 arrays (all finite attempts and
  accepted-only) for plotting the before/after R^2 distributions.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# condition -> human-readable name used by generate_fig3_noise_robustness_data.py
CONDITION_LABELS = {
    "mine_l2": "MINE + L2",
    "l2_only": "L2 only",
    "mine_no_l2": "MINE / No L2",
    "plain": "plain (no MINE, no L2)",
}


def strip_prefix(model_id: str) -> str:
    return model_id[len("model_"):] if model_id.startswith("model_") else model_id


def parse_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_logs(input_dir: Path, pattern: str):
    rows = []
    for path in sorted(glob.glob(str(input_dir / pattern))):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


def is_accepted(row, min_r2):
    """A row is accepted if flagged, and (optionally) its R^2 exceeds min_r2.

    ``min_r2`` lets us re-apply a stricter threshold than the one used during
    training (e.g. re-select at 0.997 from runs accepted at 0.99).
    """
    if str(row["accepted"]).lower() != "true":
        return False
    if min_r2 is None:
        return True
    r2 = parse_float(row["mean_r2"])
    return np.isfinite(r2) and r2 > min_r2


def aggregate(rows, min_r2=None):
    by_cond = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)

    accepted_ids = {}
    summary = []
    r2_arrays = {}
    for cond, cond_rows in by_cond.items():
        attempts = len(cond_rows)
        accepted_rows = [r for r in cond_rows if is_accepted(r, min_r2)]
        all_r2 = np.array([parse_float(r["mean_r2"]) for r in cond_rows])
        all_r2_finite = all_r2[np.isfinite(all_r2)]
        acc_r2 = np.array([parse_float(r["mean_r2"]) for r in accepted_rows])
        acc_r2 = acc_r2[np.isfinite(acc_r2)]

        # Preserve first-seen order, drop duplicate IDs if logs overlap.
        ids = list(dict.fromkeys(strip_prefix(r["model_id"]) for r in accepted_rows))
        accepted_ids[cond] = ids
        r2_arrays[f"{cond}__all"] = all_r2_finite
        r2_arrays[f"{cond}__accepted"] = acc_r2

        summary.append({
            "condition": cond,
            "label": CONDITION_LABELS.get(cond, cond),
            "attempts": attempts,
            "accepted": len(ids),
            "acceptance_rate": round(len(accepted_rows) / attempts, 4) if attempts else 0.0,
            "r2_before_mean": round(float(np.mean(all_r2_finite)), 6) if all_r2_finite.size else "",
            "r2_before_sd": round(float(np.std(all_r2_finite, ddof=1)), 6) if all_r2_finite.size > 1 else "",
            "r2_before_min": round(float(np.min(all_r2_finite)), 6) if all_r2_finite.size else "",
            "r2_after_mean": round(float(np.mean(acc_r2)), 6) if acc_r2.size else "",
            "r2_after_sd": round(float(np.std(acc_r2, ddof=1)), 6) if acc_r2.size > 1 else "",
            "r2_after_min": round(float(np.min(acc_r2)), 6) if acc_r2.size else "",
        })
    return accepted_ids, summary, r2_arrays


def write_outputs(out_dir: Path, accepted_ids, summary, r2_arrays):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "accepted_models.json").write_text(json.dumps(accepted_ids, indent=2))

    fields = ["condition", "label", "attempts", "accepted", "acceptance_rate",
              "r2_before_mean", "r2_before_sd", "r2_before_min",
              "r2_after_mean", "r2_after_sd", "r2_after_min"]
    with (out_dir / "model_selection_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    np.savez(out_dir / "model_selection_r2.npz", **r2_arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--pattern", default="training_acceptance_log_*.csv")
    parser.add_argument("--min-r2", type=float, default=None,
                        help="Re-select only models with mean R^2 above this value.")
    args = parser.parse_args()

    rows = read_logs(args.input_dir, args.pattern)
    if not rows:
        raise SystemExit(f"No acceptance logs matching {args.pattern} in {args.input_dir}")

    accepted_ids, summary, r2_arrays = aggregate(rows, args.min_r2)
    write_outputs(args.output_dir, accepted_ids, summary, r2_arrays)

    print("Model selection summary (Reviewer 1, point 2):")
    print(f"{'condition':<12}{'attempts':>9}{'accepted':>9}{'rate':>7}"
          f"{'R2_before(mean)':>17}{'R2_after(mean)':>16}")
    for s in summary:
        print(f"{s['condition']:<12}{s['attempts']:>9}{s['accepted']:>9}"
              f"{s['acceptance_rate']:>7}{str(s['r2_before_mean']):>17}{str(s['r2_after_mean']):>16}")
    print(f"\nWrote accepted_models.json, model_selection_summary.csv, model_selection_r2.npz "
          f"to {args.output_dir}")


if __name__ == "__main__":
    main()
