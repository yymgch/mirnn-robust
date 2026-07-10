# %%
"""Formal statistics for the selective-ablation effect (Reviewer 1, point 5).

For each output (Lorenz, Rossler) and each training condition (MI+L2, L2-only),
ablation is evaluated on the same models for both the primary and the
non-primary subgroup. This script quantifies the selective-damage signature:

1. gap(k) = R^2(non-primary ablated) - R^2(primary ablated), per model, with a
   t-based 95% CI at each k (large positive gap = damage is localized to the
   primary subgroup).
2. gap at the maximum k compared between conditions (MI+L2 vs L2-only) with
   Cohen's d and a Welch t-test: does MI+L2 show a larger gap than L2-only?
3. A condition x target x k interaction model
   ``R2 ~ C(condition) * C(target) * k`` fit by OLS; the three-way interaction
   tests whether the way damage depends on the ablation target and its
   magnitude differs between training conditions. (Observations are repeated
   within model; the OLS p-values are approximate and reported as such.)

Input: the per-model ablation matrices saved by
generate_fig4_dropout_robustness_data.py (``combined_results3_*.pkl``,
each a dict {suffix: (n_models, n_k)} with column 0 == k=0).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t as student_t, ttest_ind

FILES = {
    ("Lorenz", "nonprimary"): "combined_results3_1.pkl",
    ("Lorenz", "primary"): "combined_results3_1warui.pkl",
    ("Rossler", "nonprimary"): "combined_results3_2.pkl",
    ("Rossler", "primary"): "combined_results3_2warui.pkl",
}
COND = {"_m": "MI+L2", "_nm": "L2-only"}


def t_ci(values, confidence=0.95):
    values = np.asarray(values, float)
    n = len(values)
    if n < 2:
        return values.mean(), np.nan, np.nan
    half = values.std(ddof=1) / np.sqrt(n) * student_t.ppf((1 + confidence) / 2, n - 1)
    return values.mean(), values.mean() - half, values.mean() + half


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def load(input_dir):
    out = {}
    for key, fname in FILES.items():
        with (input_dir / fname).open("rb") as f:
            out[key] = pickle.load(f)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    data = load(args.input_dir)

    long_rows = []
    for output in ["Lorenz", "Rossler"]:
        nonp = data[(output, "nonprimary")]
        prim = data[(output, "primary")]
        print(f"\n================ {output} output ================")
        for suffix, cond in COND.items():
            if suffix not in nonp or suffix not in prim:
                continue
            N = np.asarray(nonp[suffix])  # (n_models, n_k)
            P = np.asarray(prim[suffix])
            n_models, n_k = N.shape
            gap = N - P  # paired per model
            print(f"\n  {cond} (n={n_models}): gap(k) = nonprimary - primary  [95% CI]")
            for k in [0, 5, 10, n_k - 1]:
                if k < n_k:
                    m, lo, hi = t_ci(gap[:, k])
                    print(f"    k={k:2d}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            # accumulate long-form for interaction model
            for tgt, arr in [("nonprimary", N), ("primary", P)]:
                for i in range(n_models):
                    for k in range(n_k):
                        long_rows.append({"R2": arr[i, k], "condition": cond,
                                          "target": tgt, "k": k, "output": output,
                                          "model": f"{cond}_{tgt}_{i}"})

        # gap at max k: compare conditions
        kmax = np.asarray(nonp["_m"]).shape[1] - 1 if "_m" in nonp else None
        if "_m" in nonp and "_nm" in nonp:
            gm = np.asarray(nonp["_m"])[:, kmax] - np.asarray(prim["_m"])[:, kmax]
            gl = np.asarray(nonp["_nm"])[:, kmax] - np.asarray(prim["_nm"])[:, kmax]
            d = cohens_d(gm, gl)
            p = ttest_ind(gm, gl, equal_var=False).pvalue
            mm, mlo, mhi = t_ci(gm)
            ml, llo, lhi = t_ci(gl)
            print(f"\n  Gap at k={kmax} (selectivity): MI+L2 {mm:+.3f}[{mlo:+.3f},{mhi:+.3f}] "
                  f"vs L2-only {ml:+.3f}[{llo:+.3f},{lhi:+.3f}]")
            print(f"    MI+L2 vs L2-only: Cohen's d={d:+.3f}, Welch p={p:.4g}  "
                  f"({'MI+L2 gap larger' if mm > ml else 'L2-only gap larger'})")

    # 3-way interaction OLS per output
    df = pd.DataFrame(long_rows)
    print("\n================ condition x target x k interaction (OLS) ================")
    print("(repeated within model; p-values approximate)")
    for output in ["Lorenz", "Rossler"]:
        sub = df[df.output == output]
        if sub.condition.nunique() < 2:
            continue
        model = smf.ols("R2 ~ C(condition) * C(target) * k", data=sub).fit()
        print(f"\n  {output}: key interaction terms")
        for name in model.params.index:
            if name.count(":") >= 1 and "condition" in name and "target" in name:
                print(f"    {name:55s} beta={model.params[name]:+.4f}  p={model.pvalues[name]:.4g}")


if __name__ == "__main__":
    main()
