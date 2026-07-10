# %%
"""Curve-level noise-robustness metrics (Reviewer 1, point 3).

Beyond the per-sigma Welch tests, this script summarizes each model's whole
R^2(sigma) degradation curve with three interpretable metrics and compares the
training conditions with effect sizes and confidence intervals:

* AUC: area under the R^2(sigma) curve (trapezoid), higher = more robust.
* slope: linear degradation slope of R^2 vs sigma (least squares), more negative
  = faster degradation.
* sigma_cross: the noise level at which R^2 first falls below a threshold
  (default 0.9 and 0.5), by linear interpolation; larger = more robust.

For each metric it reports the per-condition mean with a t-based 95% CI, and for
each pairwise comparison Cohen's d, the difference with its 95% CI, and a Welch
t-test p-value (Holm-corrected across the three comparisons).

Input: the per-model R^2 curves saved by generate_fig3_noise_robustness_data.py
(``ketteikeisu3_*.npy``, shape (n_models, n_sigma)).
"""
from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t, ttest_ind
from statsmodels.stats.multitest import multipletests

# label -> filename written by generate_fig3_noise_robustness_data.py
CONDITION_FILES = {
    "MI+L2": "ketteikeisu3_MINEあり.npy",
    "L2-only": "ketteikeisu3_MINEなし.npy",
    "unregularized": "ketteikeisu3_正則化なし.npy",
}
DEFAULT_LEVELS = np.linspace(0.0, 0.30, 31)
CROSS_THRESHOLDS = (0.9, 0.5)

# ``np.trapz`` was renamed to ``np.trapezoid`` in NumPy 2.0; support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def t_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()
    if n < 2:
        return mean, np.nan, np.nan
    half = values.std(ddof=1) / np.sqrt(n) * student_t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - half, mean + half


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def diff_ci(a, b, confidence=0.95):
    """95% CI for the difference of means (Welch)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    se = np.sqrt(a.var(ddof=1) / na + b.var(ddof=1) / nb)
    # Welch-Satterthwaite dof
    df = se**4 / ((a.var(ddof=1) / na) ** 2 / (na - 1) + (b.var(ddof=1) / nb) ** 2 / (nb - 1))
    crit = student_t.ppf((1 + confidence) / 2, df)
    d = a.mean() - b.mean()
    return d, d - crit * se, d + crit * se


def sigma_crossing(curve, levels, threshold):
    """First sigma where the curve drops below threshold (linear interp)."""
    curve = np.asarray(curve)
    below = np.where(curve < threshold)[0]
    if below.size == 0:
        return float(levels[-1])  # never crosses within the swept range
    j = below[0]
    if j == 0:
        return float(levels[0])
    x0, x1 = levels[j - 1], levels[j]
    y0, y1 = curve[j - 1], curve[j]
    if y1 == y0:
        return float(x1)
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))


def per_model_metrics(curves, levels):
    out = {"AUC": [], "slope": []}
    for thr in CROSS_THRESHOLDS:
        out[f"sigma@{thr}"] = []
    for r in curves:
        out["AUC"].append(_trapezoid(r, levels))
        out["slope"].append(np.polyfit(levels, r, 1)[0])
        for thr in CROSS_THRESHOLDS:
            out[f"sigma@{thr}"].append(sigma_crossing(r, levels, thr))
    return {k: np.asarray(v) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("noise_metrics_summary.csv"))
    args = parser.parse_args()

    levels = DEFAULT_LEVELS
    data = {}
    for label, fname in CONDITION_FILES.items():
        path = args.input_dir / fname
        if path.exists():
            data[label] = per_model_metrics(np.load(path), levels)
        else:
            print(f"  note: {fname} not found, skipping {label}")
    if not data:
        raise SystemExit("No condition files found.")

    metrics = ["AUC", "slope"] + [f"sigma@{thr}" for thr in CROSS_THRESHOLDS]
    rows = []
    print("=== Per-condition metrics (mean [95% CI]) ===")
    for m in metrics:
        print(f"\n{m}:")
        for label, md in data.items():
            mean, lo, hi = t_ci(md[m])
            n = len(md[m])
            print(f"  {label:14s} n={n:3d}  {mean:.4f}  [{lo:.4f}, {hi:.4f}]")
            rows.append({"metric": m, "comparison": f"{label} (mean)", "value": f"{mean:.5f}",
                         "ci_low": f"{lo:.5f}", "ci_high": f"{hi:.5f}", "n": n, "cohens_d": "", "p_holm": ""})

    print("\n=== Pairwise comparisons (Cohen's d, diff [95% CI], Welch p Holm-corrected per metric) ===")
    for m in metrics:
        labels = list(data.keys())
        pairs = list(combinations(labels, 2))
        pvals = []
        recs = []
        for a, b in pairs:
            va, vb = data[a][m], data[b][m]
            d = cohens_d(va, vb)
            diff, dlo, dhi = diff_ci(va, vb)
            p = ttest_ind(va, vb, equal_var=False).pvalue
            pvals.append(p)
            recs.append((a, b, d, diff, dlo, dhi))
        p_holm = multipletests(pvals, method="holm")[1] if pvals else []
        print(f"\n{m}:")
        for (a, b, d, diff, dlo, dhi), ph in zip(recs, p_holm):
            print(f"  {a} vs {b}: d={d:+.3f}, diff={diff:+.4f} [{dlo:+.4f},{dhi:+.4f}], p_holm={ph:.4g}")
            rows.append({"metric": m, "comparison": f"{a} vs {b}", "value": f"{diff:.5f}",
                         "ci_low": f"{dlo:.5f}", "ci_high": f"{dhi:.5f}", "n": "",
                         "cohens_d": f"{d:.4f}", "p_holm": f"{ph:.4g}"})

    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "comparison", "value", "ci_low", "ci_high", "n", "cohens_d", "p_holm"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
