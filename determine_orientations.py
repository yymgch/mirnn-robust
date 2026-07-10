# %%
"""Determine the subgroup-to-output orientation of each accepted model for Fig. 4.

For the ablation analysis (Fig. 4) every network is split, for each output, into a
*primary* and a *non-primary* subgroup, defined as the predefined hidden subgroup
(front = units 0..N/2-1, back = units N/2..N-1) with the higher / lower mean
absolute correlation with that output (Materials and Methods). This script
computes those correlations directly from each trained model and assigns a global
orientation label used by ``generate_fig4_dropout_robustness_data.py``:

- ``backslash``: the front subgroup is primary for the Lorenz (upper) output and
  the back subgroup is primary for the Rossler (lower) output.
- ``slash``: the reverse assignment.

The orientation is decided by which subgroup correlates more strongly with the
Lorenz output, applying the same rule to every model (including weakly
differentiated controls), as described in the manuscript.

Input:  ``accepted_models.json`` (from select_accepted_models.py) and the saved
        ``model_<id>`` weights.
Output: ``model_orientations.json`` with keys ``{backslash,slash}_<suffix>``
        (suffix _m = mine_l2, _nm = l2_only, _nr = mine_no_l2, _nmr = plain),
        consumed by generate_fig4_dropout_robustness_data.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import generate_fig4_dropout_robustness_data as fig4

# condition name in accepted_models.json -> model-id suffix used by Fig. 4
CONDITION_TO_SUFFIX = {
    "mine_l2": "_m",
    "l2_only": "_nm",
    "mine_no_l2": "_nr",
    "plain": "_nmr",
}

T0 = 200  # discard the initial transient when computing correlations


def per_neuron_output_correlation(model, x, n_samples, batch=10, t0=T0):
    """Mean absolute correlation between each hidden unit and each output channel.

    Returns an array of shape (HIDDEN_DIM, OUTPUT_DIM), averaged over samples and
    computed over time steps >= t0, using the network's reconstructed outputs.
    Processed in small batches with a single GRU pass per batch to bound memory.
    """
    acc = np.zeros((fig4.HIDDEN_DIM, 6))
    count = 0
    for start in range(0, n_samples, batch):
        xb = x[start:start + batch]
        hidden = np.asarray(model.gru(xb, training=False))      # (b, T, 100)
        out = np.asarray(model.dense1(hidden))                  # (b, T, 6)
        for s in range(hidden.shape[0]):
            h = hidden[s, t0:, :]
            o = out[s, t0:, :]
            h = (h - h.mean(0)) / (h.std(0) + 1e-12)
            o = (o - o.mean(0)) / (o.std(0) + 1e-12)
            acc += np.abs((h.T @ o) / h.shape[0])
            count += 1
    return acc / count


def orientation_for_model(model_id, x_val, n_samples):
    model = fig4.load_model(model_id, x_val)
    corr = per_neuron_output_correlation(model, x_val, n_samples)
    half = fig4.HIDDEN_DIM // 2
    # Lorenz output = front 3 channels (0:3); average over those channels.
    lorenz_corr = corr[:, 0:3].mean(axis=1)
    front_lorenz = lorenz_corr[:half].mean()
    back_lorenz = lorenz_corr[half:].mean()
    # front primary for Lorenz -> backslash, else slash.
    return "backslash" if front_lorenz >= back_lorenz else "slash"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-json", type=Path, default=Path("accepted_models.json"))
    parser.add_argument("--output", type=Path, default=Path("model_orientations.json"))
    parser.add_argument("--eval-samples", type=int, default=50,
                        help="Number of eval sequences used to estimate correlations.")
    args = parser.parse_args()

    fig4.configure_gpu_memory_growth()
    selection = json.loads(args.accepted_json.read_text())
    x_val, _ = fig4.load_evaluation_data()
    n_samples = min(args.eval_samples, x_val.shape[0])

    groups: dict[str, list[str]] = {}
    for condition, ids in selection.items():
        suffix = CONDITION_TO_SUFFIX.get(condition)
        if suffix is None:
            print(f"  skipping unknown condition '{condition}'")
            continue
        groups[f"backslash{suffix}"] = []
        groups[f"slash{suffix}"] = []
        for i, model_id in enumerate(ids, start=1):
            orientation = orientation_for_model(model_id, x_val, n_samples)
            groups[f"{orientation}{suffix}"].append(model_id)
            print(f"[{condition}] {i}/{len(ids)} model_{model_id} -> {orientation}")
        n_b = len(groups[f"backslash{suffix}"])
        n_s = len(groups[f"slash{suffix}"])
        print(f"  {condition}{suffix}: backslash={n_b}, slash={n_s}")

    args.output.write_text(json.dumps(groups, indent=2))
    print(f"\nWrote {args.output} with {sum(len(v) for v in groups.values())} models "
          f"across {len(groups)} orientation groups")


if __name__ == "__main__":
    main()
