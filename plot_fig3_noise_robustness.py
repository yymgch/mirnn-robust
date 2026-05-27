# %%
"""Plot Fig. 3 noise-robustness curves from saved experiment data."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t, ttest_ind
from statsmodels.stats.multitest import multipletests


# ================================
# Settings
# ================================
INPUT_FILES = {
    "MINE + L2": "ketteikeisu3_MINEあり.npy",
    "L2 only": "ketteikeisu3_MINEなし.npy",
    "No MINE / No L2": "ketteikeisu3_正則化なし.npy",
}
PLOT_ORDER = ["MINE + L2", "L2 only", "No MINE / No L2"]
NOISE_LEVELS = np.linspace(0.0, 0.30, 31)
OUTPUT_PATH = Path("Fig/noise_robustness_3groups.pdf")
NORMALIZED_OUTPUT_PATH = Path("Fig/noise_robustness_normalized.pdf")
SAVE_NORMALIZED_TWO_GROUP_FIGURE = True

COLORS = {
    "MINE + L2": "green",
    "L2 only": "orange",
    "No MINE / No L2": "purple",
}
MARKERS = {
    "MINE + L2": "o",
    "L2 only": "s",
    "No MINE / No L2": "D",
}
LINESTYLES = {
    "MINE + L2": "-",
    "L2 only": "--",
    "No MINE / No L2": ":",
}


# ================================
# Helpers
# ================================
def load_results(input_files):
    return {name: np.load(path) for name, path in input_files.items()}


def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(n)
    half_width = sem * t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - half_width, mean + half_width


def summarize_curves(results, order):
    means = {}
    lower = {}
    upper = {}
    for name in order:
        group_means = []
        group_lower = []
        group_upper = []
        for index in range(results[name].shape[1]):
            mean, lo, hi = mean_confidence_interval(results[name][:, index])
            group_means.append(mean)
            group_lower.append(lo)
            group_upper.append(hi)
        means[name] = np.asarray(group_means)
        lower[name] = np.asarray(group_lower)
        upper[name] = np.asarray(group_upper)
    return means, lower, upper


def baseline_normalize(results_array):
    normalized = []
    for row in results_array:
        base = row[0]
        if abs(base) < 1e-8:
            normalized.append(np.zeros_like(row))
        else:
            normalized.append(row / base)
    return np.asarray(normalized)


def print_pairwise_tests(results):
    pairs = [
        ("MINE + L2", "No MINE / No L2"),
        ("MINE + L2", "L2 only"),
        ("L2 only", "No MINE / No L2"),
    ]
    for left, right in pairs:
        p_values = []
        for index in range(results[left].shape[1]):
            _, p_value = ttest_ind(results[left][:, index], results[right][:, index], equal_var=False)
            p_values.append(p_value)
        _, corrected, _, _ = multipletests(p_values, method="holm")
        print(f"\n{left} vs {right} (Welch t-test, Holm corrected)")
        for noise_level, raw, corr in zip(NOISE_LEVELS, p_values, corrected):
            print(f"  noise={noise_level:.2f}: raw={raw:.5g}, corrected={corr:.5g}")


def plot_three_group_figure(results, output_path):
    means, lower, upper = summarize_curves(results, PLOT_ORDER)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in PLOT_ORDER:
        yerr = [means[name] - lower[name], upper[name] - means[name]]
        ax.errorbar(
            NOISE_LEVELS,
            means[name],
            yerr=yerr,
            fmt=LINESTYLES[name] + MARKERS[name],
            capsize=4,
            color=COLORS[name],
            label=name,
        )

    ax.set_xlabel("Noise Level")
    ax.set_ylabel(r"$R^2$")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"Saved figure: {output_path}")
    return fig


def plot_normalized_two_group_figure(results, output_path):
    normalized = {
        "MI minimized": baseline_normalize(results["MINE + L2"]),
        "No MI regularization": baseline_normalize(results["L2 only"]),
    }
    order = ["MI minimized", "No MI regularization"]
    colors = {"MI minimized": "green", "No MI regularization": "purple"}
    markers = {"MI minimized": "o", "No MI regularization": "D"}
    linestyles = {"MI minimized": "-", "No MI regularization": "--"}

    means, lower, upper = summarize_curves(normalized, order)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in order:
        yerr = [means[name] - lower[name], upper[name] - means[name]]
        ax.errorbar(
            NOISE_LEVELS,
            means[name],
            yerr=yerr,
            fmt=linestyles[name] + markers[name],
            capsize=4,
            color=colors[name],
            label=name,
        )

    ax.set_xlabel("Noise Level")
    ax.set_ylabel(r"Normalized $R^2$")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"Saved figure: {output_path}")
    return fig


# ================================
# Main
# ================================
def main():
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    results = load_results(INPUT_FILES)
    for name in PLOT_ORDER:
        print(f"{name}: shape={results[name].shape}, baseline mean={results[name][:, 0].mean():.5f}")

    print_pairwise_tests(results)
    plot_three_group_figure(results, OUTPUT_PATH)

    if SAVE_NORMALIZED_TWO_GROUP_FIGURE:
        plot_normalized_two_group_figure(results, NORMALIZED_OUTPUT_PATH)

    plt.show()


if __name__ == "__main__":
    main()
