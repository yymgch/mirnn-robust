# %%
"""Mechanism check across training conditions (Reviewer 1, point 6).

To support the interpretation that MI minimization reduces inter-subgroup
dependence (and thus cross-talk), this script aggregates, per condition, a set
of mechanism indicators over the accepted models:

* inter_subgroup_corr: mean absolute correlation between the two predefined
  hidden subgroups (off-diagonal block of the |corr| matrix of GRU activities).
  Lower = weaker statistical dependence between populations.
* M (functional modularity), S (correlation separation index),
  D (readout-weight separation index): the indicators saved during training.
* final_MI: the final MINE estimate (available for MINE-trained conditions).
* readout_block_ratio: ratio of each output channel's readout weight mass on its
  primary vs non-primary subgroup (higher = more segregated readout).

Inputs: ``accepted_models.json`` plus the saved per-model arrays
(``M_<id>.npy``, ``S_<id>.npy``, ``D_<id>.npy``, ``mi_<id>.npy``) and the model
weights ``model_<id>``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

import generate_fig4_dropout_robustness_data as fig4

HALF = fig4.HIDDEN_DIM // 2


def t_ci(values, confidence=0.95):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan
    if n < 2:
        return values.mean(), np.nan, np.nan
    half = values.std(ddof=1) / np.sqrt(n) * student_t.ppf((1 + confidence) / 2, n - 1)
    return values.mean(), values.mean() - half, values.mean() + half


def final_value(path):
    if not path.exists():
        return np.nan
    return float(np.asarray(np.load(path)).reshape(-1)[-1])


def inter_subgroup_corr(model, x, n_samples):
    """Mean |corr| between the front and back subgroups of GRU activities."""
    vals = []
    for start in range(0, n_samples, 10):
        hidden = np.asarray(model.gru(x[start:start + 10], training=False))  # (b,T,100)
        for s in range(hidden.shape[0]):
            c = np.abs(np.corrcoef(hidden[s].T))  # (100,100) over time
            vals.append(np.nanmean(c[:HALF, HALF:]))  # front x back block
    return float(np.mean(vals))


def readout_block_ratio(model):
    """Mean over output channels of (weight mass on primary subgroup)/(total)."""
    w = np.abs(np.asarray(model.dense1.get_weights()[0]))  # (100, 6)
    front = w[:HALF, :].sum(axis=0)
    back = w[HALF:, :].sum(axis=0)
    primary = np.maximum(front, back)
    total = front + back
    return float(np.mean(primary / np.where(total > 0, total, 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--accepted-json", type=Path, default=Path("accepted_models.json"))
    parser.add_argument("--eval-samples", type=int, default=20)
    args = parser.parse_args()

    fig4.configure_gpu_memory_growth()
    selection = json.loads(args.accepted_json.read_text())
    x_val, _ = fig4.load_evaluation_data()
    n_samples = min(args.eval_samples, x_val.shape[0])

    metrics = ["inter_subgroup_corr", "M", "S", "D", "final_MI", "readout_block_ratio"]
    results = {cond: {m: [] for m in metrics} for cond in selection}

    for cond, ids in selection.items():
        for i, mid in enumerate(ids, 1):
            r = results[cond]
            r["M"].append(final_value(args.input_dir / f"M_{mid}.npy"))
            r["S"].append(final_value(args.input_dir / f"S_{mid}.npy"))
            r["D"].append(final_value(args.input_dir / f"D_{mid}.npy"))
            r["final_MI"].append(final_value(args.input_dir / f"mi_{mid}.npy"))
            model = fig4.load_model(mid, x_val)
            r["inter_subgroup_corr"].append(inter_subgroup_corr(model, x_val, n_samples))
            r["readout_block_ratio"].append(readout_block_ratio(model))
            print(f"[{cond}] {i}/{len(ids)} model_{mid} done", flush=True)

    print("\n=== Mechanism indicators per condition (mean [95% CI]) ===")
    print(f"{'metric':<22}" + "".join(f"{c:>22}" for c in selection))
    for m in metrics:
        cells = []
        for cond in selection:
            mean, lo, hi = t_ci(results[cond][m])
            cells.append(f"{mean:.3f}[{lo:.3f},{hi:.3f}]" if np.isfinite(mean) else "n/a")
        print(f"{m:<22}" + "".join(f"{c:>22}" for c in cells))


if __name__ == "__main__":
    main()
