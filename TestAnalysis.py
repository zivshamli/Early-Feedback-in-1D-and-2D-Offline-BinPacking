import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILES = {
    "MC": "MCResult/MC_test_results_seed_*.csv",
    "TD_EF": "TDWithEF/TD_EF_test_results_seed_*.csv",
}

OUTPUT_DIR = "test_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD TEST FILES
# ============================================================

all_results = []

for model, pattern in MODEL_FILES.items():

    files = sorted(glob.glob(pattern))

    if not files:
        print(f"WARNING: No files found for {model}")
        continue

    print(f"\n{model}:")
    print(f"Found {len(files)} test files")

    for filepath in files:

        filename = os.path.basename(filepath)

        print(f"  {filename}")

        # ----------------------------------------------------
        # Extract seed
        # ----------------------------------------------------

        try:
            seed = int(
                filename.split("seed_")[1]
                .split(".csv")[0]
            )

        except (IndexError, ValueError):

            raise ValueError(
                f"Could not extract seed from filename: "
                f"{filename}"
            )

        # ----------------------------------------------------
        # Load CSV
        # ----------------------------------------------------

        df = pd.read_csv(filepath)

        # ----------------------------------------------------
        # Check columns
        # ----------------------------------------------------

        required_columns = [
            "instance",
            "bins",
            "optimal_bins",
            "utilization",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:

            raise ValueError(
                f"{filename} is missing columns: "
                f"{missing_columns}"
            )

        # ----------------------------------------------------
        # Remove invalid rows
        # ----------------------------------------------------

        df = df.dropna(
            subset=[
                "bins",
                "optimal_bins",
                "utilization"
            ]
        ).copy()

        # ----------------------------------------------------
        # Calculate Optimality Gap
        #
        # Gap = (bins - optimal_bins) / optimal_bins
        # ----------------------------------------------------

        df["optimality_gap"] = (
            (df["bins"] - df["optimal_bins"])
            / df["optimal_bins"]
        )

        # ----------------------------------------------------
        # Calculate the 4 requested metrics
        # ----------------------------------------------------

        mean_utilization = (
            df["utilization"].mean()
        )

        std_utilization = (
            df["utilization"].std()
        )

        mean_optimality_gap = (
            df["optimality_gap"].mean()
        )

        std_optimality_gap = (
            df["optimality_gap"].std()
        )

        # ----------------------------------------------------
        # Save result for this seed
        # ----------------------------------------------------

        all_results.append({

            "model": model,

            "seed": seed,

            "mean_utilization":
                mean_utilization,

            "std_utilization":
                std_utilization,

            "mean_optimality_gap":
                mean_optimality_gap,

            "std_optimality_gap":
                std_optimality_gap,
        })


# ============================================================
# CHECK RESULTS
# ============================================================

if not all_results:

    raise RuntimeError(
        "No test files were found."
    )


# ============================================================
# PER-SEED RESULTS
# ============================================================

per_seed = pd.DataFrame(
    all_results
)

per_seed = (
    per_seed
    .sort_values(
        ["model", "seed"]
    )
    .reset_index(drop=True)
)


print("\n")
print("=" * 100)
print("TEST METRICS PER SEED")
print("=" * 100)

print(
    per_seed.to_string(
        index=False
    )
)


per_seed.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_metrics_per_seed.csv"
    ),
    index=False
)


# ============================================================
# MODEL COMPARISON
#
# Mean and STD across seeds
# ============================================================

model_summary = (
    per_seed
    .groupby("model")
    .agg(

        # ----------------------------------------------------
        # Mean Utilization
        # ----------------------------------------------------

        mean_utilization=(
            "mean_utilization",
            "mean"
        ),

        # ----------------------------------------------------
        # STD across seeds
        # ----------------------------------------------------

        utilization_between_seed_std=(
            "mean_utilization",
            "std"
        ),

        # ----------------------------------------------------
        # Mean Optimality Gap
        # ----------------------------------------------------

        mean_optimality_gap=(
            "mean_optimality_gap",
            "mean"
        ),

        # ----------------------------------------------------
        # STD across seeds
        # ----------------------------------------------------

        optimality_gap_between_seed_std=(
            "mean_optimality_gap",
            "std"
        ),
    )
    .reset_index()
)


print("\n")
print("=" * 100)
print("TEST MODEL COMPARISON")
print("MEAN ± STD ACROSS SEEDS")
print("=" * 100)

print(
    model_summary.to_string(
        index=False
    )
)


model_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_model_comparison.csv"
    ),
    index=False
)


# ============================================================
# FORMATTED SUMMARY
# ============================================================

formatted_summary = []

for _, row in model_summary.iterrows():

    formatted_summary.append({

        "Model":
            row["model"],

        "Mean Utilization":
            f'{row["mean_utilization"]:.4f} ± '
            f'{row["utilization_between_seed_std"]:.4f}',

        "Mean Optimality Gap":
            f'{row["mean_optimality_gap"]:.4f} ± '
            f'{row["optimality_gap_between_seed_std"]:.4f}',
    })


formatted_summary = pd.DataFrame(
    formatted_summary
)


print("\n")
print("=" * 100)
print("FINAL SUMMARY")
print("=" * 100)

print(
    formatted_summary.to_string(
        index=False
    )
)


formatted_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_final_summary.csv"
    ),
    index=False
)


# ============================================================
# GRAPH 1
# MEAN UTILIZATION ± STD ACROSS SEEDS
# ============================================================

plt.figure(figsize=(8, 6))

x = np.arange(
    len(model_summary)
)

plt.bar(
    x,
    model_summary["mean_utilization"],
    yerr=model_summary[
        "utilization_between_seed_std"
    ],
    capsize=6
)

plt.xticks(
    x,
    model_summary["model"]
)

plt.ylabel("Mean Utilization")

plt.title(
    "Test Mean Utilization ± STD Across Seeds"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "test_mean_utilization.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# GRAPH 2
# MEAN OPTIMALITY GAP ± STD ACROSS SEEDS
# ============================================================

plt.figure(figsize=(8, 6))

x = np.arange(
    len(model_summary)
)

plt.bar(
    x,
    model_summary["mean_optimality_gap"],
    yerr=model_summary[
        "optimality_gap_between_seed_std"
    ],
    capsize=6
)

plt.xticks(
    x,
    model_summary["model"]
)

plt.ylabel("Mean Optimality Gap")

plt.title(
    "Test Mean Optimality Gap ± STD Across Seeds"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "test_mean_optimality_gap.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# GRAPH 3
# STD UTILIZATION WITHIN TEST INSTANCES
# ============================================================

plt.figure(figsize=(8, 6))

x = np.arange(
    len(per_seed)
)

plt.bar(
    x,
    per_seed["std_utilization"]
)

labels = [
    f'{model}\nSeed {seed}'
    for model, seed
    in zip(
        per_seed["model"],
        per_seed["seed"]
    )
]

plt.xticks(
    x,
    labels
)

plt.ylabel("STD Utilization")

plt.title(
    "Test Utilization STD per Seed"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "test_std_utilization_per_seed.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# GRAPH 4
# STD OPTIMALITY GAP WITHIN TEST INSTANCES
# ============================================================

plt.figure(figsize=(8, 6))

x = np.arange(
    len(per_seed)
)

plt.bar(
    x,
    per_seed["std_optimality_gap"]
)

labels = [
    f'{model}\nSeed {seed}'
    for model, seed
    in zip(
        per_seed["model"],
        per_seed["seed"]
    )
]

plt.xticks(
    x,
    labels
)

plt.ylabel("STD Optimality Gap")

plt.title(
    "Test Optimality Gap STD per Seed"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "test_std_optimality_gap_per_seed.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 100)
print("TEST ANALYSIS COMPLETED")
print("=" * 100)

print(
    f"Output directory: {OUTPUT_DIR}"
)

