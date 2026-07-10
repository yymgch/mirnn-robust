# %%
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


# %%
# ================================
# Settings
# ================================
INPUT_DIR = Path(".")
OUTPUT_PATH = Path("Fig/dropout_robustness_primary_nonprimary.pdf")
OUTPUT_PATH_RIGHT_LEGEND = Path("Fig/dropout_robustness_primary_nonprimary_right_legend.pdf")

INPUT_FILES = {
    "upper_nonprimary": "combined_results3_1.pkl",
    "lower_nonprimary": "combined_results3_2.pkl",
    "upper_primary": "combined_results3_1warui.pkl",
    "lower_primary": "combined_results3_2warui.pkl",
}

GROUP_KEY = "_m"       # MINE + L2
L2_ONLY_KEY = "_nm"    # L2 only


# %%
# ================================
# Helper functions
# ================================
def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def load_saved_results(input_dir, input_files):
    return {
        name: load_pickle(input_dir / filename)
        for name, filename in input_files.items()
    }



# %%
# ================================
# Load saved data
# ================================
results = load_saved_results(INPUT_DIR, INPUT_FILES)

for name, data in results.items():
    print(name, list(data.keys()))


# %%
# ================================
# Prepare curves
# ================================
upper_primary = results["upper_primary"][GROUP_KEY]
lower_primary = results["lower_primary"][GROUP_KEY]
upper_nonprimary = results["upper_nonprimary"][GROUP_KEY]
lower_nonprimary = results["lower_nonprimary"][GROUP_KEY]

upper_l2_primary = results["upper_primary"][L2_ONLY_KEY]
lower_l2_primary = results["lower_primary"][L2_ONLY_KEY]
upper_l2_nonprimary = results["upper_nonprimary"][L2_ONLY_KEY]
lower_l2_nonprimary = results["lower_nonprimary"][L2_ONLY_KEY]


# %%
# ================================
# Plot helpers
# ================================
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def mean_ci(data, confidence=0.95):
    """Across-model mean and t-based 95% CI half-width per column (matches Fig. 3)."""
    data = np.asarray(data, dtype=float)
    n = data.shape[0]
    mean = data.mean(axis=0)
    sem = data.std(axis=0, ddof=1) / np.sqrt(n)
    half_width = sem * t.ppf((1 + confidence) / 2, n - 1)
    return mean, half_width


def plot_panel(ax, mine_primary, mine_nonprimary, l2_primary, l2_nonprimary, title):
    # Each curve is a per-model matrix of shape (n_models, n_k) with column 0 = k=0.
    ks = np.arange(0, np.asarray(mine_primary).shape[1])
    curves = [
        ("MINE+L2 primary ablated", mine_primary, "#007385db", "o", "-"),
        ("MINE+L2 non-primary ablated", mine_nonprimary, "#059c0d", "s", "-"),
        ("L2 only primary ablated", l2_primary, "#a02c53", "^", "--"),
        ("L2 only non-primary ablated", l2_nonprimary, "#ba00df", "D", "--"),
    ]

    for label, curve, color, marker, linestyle in curves:
        mean, half_width = mean_ci(curve)
        ax.plot(
            ks,
            mean,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=4,
            label=label,
        )
        ax.fill_between(
            ks, mean - half_width, mean + half_width, color=color, alpha=0.16, linewidth=0
        )

    ax.set_title(title)
    ax.set_xlabel("Number of ablated units")
    ax.set_ylabel(r"$R^2$")
    ax.set_xticks(ks[::2])
    ax.set_xlim(ks[0], ks[-1])
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)


# %%
# ================================
# Make figures
# ================================
def make_figure(legend_position="top"):
    if legend_position == "right":
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.0), sharey=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)

    plot_panel(
        axes[0],
        upper_primary,
        upper_nonprimary,
        upper_l2_primary,
        upper_l2_nonprimary,
        "Lorenz output",
    )
    plot_panel(
        axes[1],
        lower_primary,
        lower_nonprimary,
        lower_l2_primary,
        lower_l2_nonprimary,
        "Rössler output",
    )

    axes[0].text(-0.12, 1.04, "A", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.12, 1.04, "B", transform=axes[1].transAxes, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    if legend_position == "right":
        fig.legend(
            handles,
            labels,
            loc="upper left",
            ncol=1,
            frameon=False,
            bbox_to_anchor=(0.79, 0.82),
        )
        fig.tight_layout(rect=(0, 0, 0.77, 1))
    else:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1.08),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88))

    return fig


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

fig = make_figure(legend_position="top")
fig.savefig(OUTPUT_PATH, bbox_inches="tight")
print(f"Saved figure: {OUTPUT_PATH}")

fig_right = make_figure(legend_position="right")
fig_right.savefig(OUTPUT_PATH_RIGHT_LEGEND, bbox_inches="tight")
print(f"Saved figure: {OUTPUT_PATH_RIGHT_LEGEND}")

plt.show()

# %%
