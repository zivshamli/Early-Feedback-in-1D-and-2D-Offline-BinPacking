import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILES = {
    "MC": "MCResult/MC_validation_history_seed_*.csv",
    "TD": "TDResult/TD_validation_history_seed_*.csv",
    "TD_EF": "TDWithEF/TD_EF_validation_history_seed_*.csv",
}

OUTPUT_DIR = "validation_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD VALIDATION FILES
# ============================================================

all_data = []

for model, pattern in MODEL_FILES.items():

    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        print(f"WARNING: No validation files found for {model}")
        continue

    print(f"\n{model}")
    print(f"Found {len(files)} files")

    for filepath in files:

        filename = os.path.basename(filepath)

        print(f"  Loading: {filename}")

        # ----------------------------------------------------
        # Extract seed
        # Example:
        # MC_validation_history_seed_1.csv
        # ----------------------------------------------------

        if "seed_" not in filename:
            raise ValueError(
                f"Cannot extract seed from filename: {filename}"
            )

        seed = int(
            filename.split("seed_")[1]
            .split(".csv")[0]
        )

        df = pd.read_csv(filepath)

        # ----------------------------------------------------
        # Required columns
        # ----------------------------------------------------

        required_columns = [
            "episode",
            "utilization",
            "optimality_gap",
        ]

        missing = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{filename} is missing columns: {missing}"
            )

        df = (
            df[
                [
                    "episode",
                    "utilization",
                    "optimality_gap",
                ]
            ]
            .sort_values("episode")
            .reset_index(drop=True)
        )

        # Add metadata
        df["model"] = model
        df["seed"] = seed

        all_data.append(df)


# ============================================================
# COMBINE DATA
# ============================================================

if len(all_data) == 0:
    raise RuntimeError(
        "No validation CSV files were found."
    )

data = pd.concat(
    all_data,
    ignore_index=True
)


print("\n")
print("=" * 80)
print("LOADED VALIDATION DATA")
print("=" * 80)

print(
    data
    .groupby(["model", "seed"])
    .size()
)


# ============================================================
# PER-SEED SUMMARY
# ============================================================

# Best validation performance for each seed
#
# Higher utilization = better
# Lower optimality gap = better

best_utilization = (
    data
    .loc[
        data.groupby(["model", "seed"])["utilization"]
        .idxmax()
    ]
    .copy()
)

best_utilization = best_utilization.rename(
    columns={
        "episode": "best_utilization_episode",
        "utilization": "best_utilization",
    }
)


best_gap = (
    data
    .loc[
        data.groupby(["model", "seed"])["optimality_gap"]
        .idxmin()
    ]
    .copy()
)

best_gap = best_gap.rename(
    columns={
        "episode": "best_gap_episode",
        "optimality_gap": "best_optimality_gap",
    }
)


# ------------------------------------------------------------
# Merge best results
# ------------------------------------------------------------

per_seed_summary = pd.merge(
    best_utilization[
        [
            "model",
            "seed",
            "best_utilization",
            "best_utilization_episode",
        ]
    ],
    best_gap[
        [
            "model",
            "seed",
            "best_optimality_gap",
            "best_gap_episode",
        ]
    ],
    on=["model", "seed"],
)


# ============================================================
# FINAL VALIDATION PERFORMANCE
# ============================================================

last_rows = (
    data
    .sort_values("episode")
    .groupby(["model", "seed"])
    .tail(1)
    .copy()
)

last_rows = last_rows.rename(
    columns={
        "utilization": "final_utilization",
        "optimality_gap": "final_optimality_gap",
        "episode": "final_episode",
    }
)


per_seed_summary = pd.merge(
    per_seed_summary,
    last_rows[
        [
            "model",
            "seed",
            "final_episode",
            "final_utilization",
            "final_optimality_gap",
        ]
    ],
    on=["model", "seed"],
)


# ============================================================
# PRINT PER-SEED RESULTS
# ============================================================

print("\n")
print("=" * 100)
print("VALIDATION RESULTS PER SEED")
print("=" * 100)

print(
    per_seed_summary.to_string(
        index=False
    )
)


per_seed_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "validation_per_seed.csv"
    ),
    index=False
)


# ============================================================
# MEAN ± STD ACROSS SEEDS
# ============================================================

model_summary = (
    per_seed_summary
    .groupby("model")
    .agg(

        best_utilization_mean=(
            "best_utilization",
            "mean"
        ),

        best_utilization_std=(
            "best_utilization",
            "std"
        ),

        best_gap_mean=(
            "best_optimality_gap",
            "mean"
        ),

        best_gap_std=(
            "best_optimality_gap",
            "std"
        ),

        best_utilization_episode_mean=(
            "best_utilization_episode",
            "mean"
        ),

        best_utilization_episode_std=(
            "best_utilization_episode",
            "std"
        ),

        best_gap_episode_mean=(
            "best_gap_episode",
            "mean"
        ),

        best_gap_episode_std=(
            "best_gap_episode",
            "std"
        ),

        final_utilization_mean=(
            "final_utilization",
            "mean"
        ),

        final_utilization_std=(
            "final_utilization",
            "std"
        ),

        final_gap_mean=(
            "final_optimality_gap",
            "mean"
        ),

        final_gap_std=(
            "final_optimality_gap",
            "std"
        ),
    )
    .reset_index()
)


print("\n")
print("=" * 100)
print("VALIDATION MODEL COMPARISON")
print("MEAN +/- STD ACROSS SEEDS")
print("=" * 100)

print(
    model_summary.to_string(
        index=False
    )
)


model_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "validation_model_summary.csv"
    ),
    index=False
)


# ============================================================
# VALIDATION CURVES
# ============================================================

# At every checkpoint, calculate:
#
# Mean across seeds
# STD across seeds
#
# for each model.

curve_summary = (
    data
    .groupby(
        [
            "model",
            "episode"
        ]
    )
    .agg(

        utilization_mean=(
            "utilization",
            "mean"
        ),

        utilization_std=(
            "utilization",
            "std"
        ),

        optimality_gap_mean=(
            "optimality_gap",
            "mean"
        ),

        optimality_gap_std=(
            "optimality_gap",
            "std"
        ),
    )
    .reset_index()
)


curve_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "validation_curves_mean_std.csv"
    ),
    index=False
)


# ============================================================
# PLOT VALIDATION UTILIZATION
# ============================================================

plt.figure(figsize=(11, 6))

for model in curve_summary["model"].unique():

    model_df = curve_summary[
        curve_summary["model"] == model
    ]

    x = model_df["episode"]
    y = model_df["utilization_mean"]
    std = model_df["utilization_std"]

    plt.plot(
        x,
        y,
        marker="o",
        label=model
    )

    plt.fill_between(
        x,
        y - std,
        y + std,
        alpha=0.15
    )


plt.xlabel("Training Episode")
plt.ylabel("Validation Utilization")
plt.title(
    "Validation Utilization — Mean ± STD Across Seeds"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "validation_utilization.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# PLOT VALIDATION OPTIMALITY GAP
# ============================================================

plt.figure(figsize=(11, 6))

for model in curve_summary["model"].unique():

    model_df = curve_summary[
        curve_summary["model"] == model
    ]

    x = model_df["episode"]
    y = model_df["optimality_gap_mean"]
    std = model_df["optimality_gap_std"]

    plt.plot(
        x,
        y,
        marker="o",
        label=model
    )

    plt.fill_between(
        x,
        y - std,
        y + std,
        alpha=0.15
    )


plt.xlabel("Training Episode")
plt.ylabel("Validation Optimality Gap")
plt.title(
    "Validation Optimality Gap — Mean ± STD Across Seeds"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "validation_optimality_gap.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# BEST VALIDATION UTILIZATION
# ============================================================

plt.figure(figsize=(9, 6))

models = model_summary["model"]
means = model_summary["best_utilization_mean"]
stds = model_summary["best_utilization_std"]

x = np.arange(len(models))

plt.bar(
    x,
    means,
    yerr=stds,
    capsize=5
)

plt.xticks(
    x,
    models
)

plt.ylabel("Best Validation Utilization")
plt.title(
    "Best Validation Utilization — Mean ± STD Across Seeds"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "best_validation_utilization.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# BEST VALIDATION OPTIMALITY GAP
# ============================================================

plt.figure(figsize=(9, 6))

models = model_summary["model"]
means = model_summary["best_gap_mean"]
stds = model_summary["best_gap_std"]

x = np.arange(len(models))

plt.bar(
    x,
    means,
    yerr=stds,
    capsize=5
)

plt.xticks(
    x,
    models
)

plt.ylabel("Best Validation Optimality Gap")
plt.title(
    "Best Validation Optimality Gap — Mean ± STD Across Seeds"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "best_validation_gap.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# CONVERGENCE SPEED
# ============================================================

print("\n")
print("=" * 100)
print("CONVERGENCE / BEST VALIDATION EPISODE")
print("=" * 100)

for _, row in model_summary.iterrows():

    print(
        f"\n{row['model']}"
    )

    print(
        f"Best Utilization: "
        f"{row['best_utilization_mean']:.6f} "
        f"+/- "
        f"{row['best_utilization_std']:.6f}"
    )

    print(
        f"Episode of Best Utilization: "
        f"{row['best_utilization_episode_mean']:.1f} "
        f"+/- "
        f"{row['best_utilization_episode_std']:.1f}"
    )

    print(
        f"Best Optimality Gap: "
        f"{row['best_gap_mean']:.6f} "
        f"+/- "
        f"{row['best_gap_std']:.6f}"
    )

    print(
        f"Episode of Best Gap: "
        f"{row['best_gap_episode_mean']:.1f} "
        f"+/- "
        f"{row['best_gap_episode_std']:.1f}"
    )


# ============================================================
# EARLY FEEDBACK IMPROVEMENT
# ============================================================

lookup = model_summary.set_index("model")


def gap_improvement(
    baseline_gap,
    method_gap
):
    return (
        (baseline_gap - method_gap)
        / abs(baseline_gap)
        * 100
    )


def utilization_improvement(
    baseline_util,
    method_util
):
    return (
        (method_util - baseline_util)
        / abs(baseline_util)
        * 100
    )


# ------------------------------------------------------------
# TD_EF vs MC
# ------------------------------------------------------------

if "MC" in lookup.index and "TD_EF" in lookup.index:

    mc_gap = lookup.loc[
        "MC",
        "best_gap_mean"
    ]

    ef_gap = lookup.loc[
        "TD_EF",
        "best_gap_mean"
    ]

    mc_util = lookup.loc[
        "MC",
        "best_utilization_mean"
    ]

    ef_util = lookup.loc[
        "TD_EF",
        "best_utilization_mean"
    ]

    print("\n")
    print("=" * 80)
    print("TD + EARLY FEEDBACK vs MC")
    print("=" * 80)

    print(
        f"Optimality Gap improvement: "
        f"{gap_improvement(mc_gap, ef_gap):.2f}%"
    )

    print(
        f"Utilization improvement: "
        f"{utilization_improvement(mc_util, ef_util):.2f}%"
    )


# ------------------------------------------------------------
# TD_EF vs TD
# ------------------------------------------------------------

if "TD" in lookup.index and "TD_EF" in lookup.index:

    td_gap = lookup.loc[
        "TD",
        "best_gap_mean"
    ]

    ef_gap = lookup.loc[
        "TD_EF",
        "best_gap_mean"
    ]

    td_util = lookup.loc[
        "TD",
        "best_utilization_mean"
    ]

    ef_util = lookup.loc[
        "TD_EF",
        "best_utilization_mean"
    ]

    print("\n")
    print("=" * 80)
    print("TD + EARLY FEEDBACK vs TD")
    print("=" * 80)

    print(
        f"Optimality Gap improvement: "
        f"{gap_improvement(td_gap, ef_gap):.2f}%"
    )

    print(
        f"Utilization improvement: "
        f"{utilization_improvement(td_util, ef_util):.2f}%"
    )


print("\n")
print("=" * 80)
print("VALIDATION ANALYSIS COMPLETED")
print("=" * 80)

print(
    f"\nResults saved in: {OUTPUT_DIR}/"
)
