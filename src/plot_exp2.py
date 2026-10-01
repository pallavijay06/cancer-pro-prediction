import os

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
)

INPUT_FILE = os.path.join(
    RESULTS_DIR,
    "exp2_ablation_results.csv",
)

EXP1_FILE = os.path.join(
    RESULTS_DIR,
    "exp1_M1_y_drop.csv",
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "exp2_ablation_auroc.png",
)


# ---------------------------------------------------------
# Load Experiment 2 results
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)


# ---------------------------------------------------------
# Load full-model AUROC from Experiment 1
# ---------------------------------------------------------

exp1 = pd.read_csv(EXP1_FILE)

full_model_row = exp1[
    exp1["model"].str.lower() == "lasso"
]

if full_model_row.empty:
    raise ValueError(
        "Could not find lasso in exp1_M1_y_drop.csv"
    )

full_auroc = float(
    full_model_row.iloc[0]["auc_mean"]
)


# ---------------------------------------------------------
# Labels
# ---------------------------------------------------------

labels = [
    "Full model",
    "Remove demographics",
    "Remove disease",
    "Remove treatment",
    "Remove psychological",
    "Remove baseline QoL",
    "Baseline QoL only",
    "Everything except baseline QoL",
]


# ---------------------------------------------------------
# Build plotting data
# ---------------------------------------------------------

plot_auroc = [
    full_auroc,
    *df["auroc_mean"].tolist(),
]

plot_low = [
    full_auroc,
    *df["bootstrap_ci_low"].tolist(),
]

plot_high = [
    full_auroc,
    *df["bootstrap_ci_high"].tolist(),
]

# Convert CI to asymmetric error bars.
lower_error = [
    auroc - low
    for auroc, low in zip(
        plot_auroc,
        plot_low,
    )
]

upper_error = [
    high - auroc
    for auroc, high in zip(
        plot_auroc,
        plot_high,
    )
]


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(11, 6)
)

x = range(len(labels))

ax.errorbar(
    x,
    plot_auroc,
    yerr=[
        lower_error,
        upper_error,
    ],
    fmt="o",
    capsize=5,
)

ax.axhline(
    full_auroc,
    linestyle="--",
    linewidth=1,
)

ax.set_xticks(
    list(x)
)

ax.set_xticklabels(
    labels,
    rotation=30,
    ha="right",
)

ax.set_ylabel(
    "AUROC"
)

ax.set_title(
    "Experiment 2: Feature-Group Ablation"
)

ax.set_ylim(
    0.45,
    0.85
)

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    f"Saved figure -> {OUTPUT_FILE}"
)