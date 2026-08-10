import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILES = {
    "MC": "MCResult/MC_train_history_seed_*.csv",
    "TD_EF": "TDWithEF/TD_EF_train_history_seed_*.csv",
}

WINDOW_SIZE = 200

OUTPUT_DIR = "train_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD ALL TRAIN FILES
# ============================================================

all_data = []

for model, pattern in MODEL_FILES.items():

    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        print(f"WARNING: No files found for {model}")
        continue

    print(f"\n{model}:")
    print(f"Found {len(files)} files")

    for filepath in files:

        filename = os.path.basename(filepath)

        print(f"  Loading: {filename}")

        # ----------------------------------------------------
        # Extract seed from filename
        #
        # Example:
        # MC_train_history_seed_1.csv
        # TD_EF_train_history_seed_1.csv
        # ----------------------------------------------------

        seed = filename.split("seed_")[1].split(".csv")[0]

        df = pd.read_csv(filepath)

        # ----------------------------------------------------
        # Check required columns
        # ----------------------------------------------------

        required_columns = [
            "episode",
            "reward",
            "bins",
            "optimal_bins",
            "utilization",
            "actor_loss",
            "critic_loss",
            "gradient_variance",
            "gradient_norm",
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

        # ----------------------------------------------------
        # Calculate Optimality Gap
        # ----------------------------------------------------

        df["optimality_gap"] = (
            (df["bins"] - df["optimal_bins"])
            / df["optimal_bins"]
        )

        df["model"] = model
        df["seed"] = int(seed)

        df = (
            df
            .sort_values("episode")
            .reset_index(drop=True)
        )

        all_data.append(df)


# ============================================================
# COMBINE ALL DATA
# ============================================================

if len(all_data) == 0:
    raise RuntimeError(
        "No CSV files were found."
    )

data = pd.concat(
    all_data,
    ignore_index=True
)


print("\nTotal rows:", len(data))

print("\nLoaded data:")
print(
    data[
        [
            "model",
            "seed",
            "episode"
        ]
    ]
    .groupby(["model", "seed"])
    .size()
)


# ============================================================
# OVERALL STATISTICS PER SEED
# ============================================================

seed_statistics = (
    data
    .groupby(["model", "seed"])
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

        actor_loss_mean=(
            "actor_loss",
            "mean"
        ),

        critic_loss_mean=(
            "critic_loss",
            "mean"
        ),

        gradient_variance_mean=(
            "gradient_variance",
            "mean"
        ),

        gradient_variance_std=(
            "gradient_variance",
            "std"
        ),

        gradient_norm_mean=(
            "gradient_norm",
            "mean"
        ),

        gradient_norm_std=(
            "gradient_norm",
            "std"
        ),
    )
    .reset_index()
)


print("\n")
print("=" * 100)
print("PER-SEED TRAIN RESULTS")
print("=" * 100)

print(
    seed_statistics.to_string(
        index=False
    )
)


seed_statistics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "per_seed_results.csv"
    ),
    index=False
)


# ============================================================
# MEAN ± STD ACROSS SEEDS
# ============================================================

model_statistics = (
    seed_statistics
    .groupby("model")
    .agg(

        utilization_mean=(
            "utilization_mean",
            "mean"
        ),

        utilization_seed_std=(
            "utilization_mean",
            "std"
        ),

        optimality_gap_mean=(
            "optimality_gap_mean",
            "mean"
        ),

        optimality_gap_seed_std=(
            "optimality_gap_mean",
            "std"
        ),

        actor_loss_mean=(
            "actor_loss_mean",
            "mean"
        ),

        actor_loss_seed_std=(
            "actor_loss_mean",
            "std"
        ),

        critic_loss_mean=(
            "critic_loss_mean",
            "mean"
        ),

        critic_loss_seed_std=(
            "critic_loss_mean",
            "std"
        ),

        gradient_variance_mean=(
            "gradient_variance_mean",
            "mean"
        ),

        gradient_variance_seed_std=(
            "gradient_variance_mean",
            "std"
        ),

        gradient_norm_mean=(
            "gradient_norm_mean",
            "mean"
        ),

        gradient_norm_seed_std=(
            "gradient_norm_mean",
            "std"
        ),
    )
    .reset_index()
)


print("\n")
print("=" * 100)
print("MODEL COMPARISON — MEAN ± STD ACROSS SEEDS")
print("=" * 100)

print(
    model_statistics.to_string(
        index=False
    )
)


model_statistics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    ),
    index=False
)


# ============================================================
# WINDOW ANALYSIS
# ============================================================

data["window"] = (
    data["episode"] // WINDOW_SIZE
)


window_statistics = (
    data
    .groupby(
        [
            "model",
            "seed",
            "window"
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

        actor_loss_mean=(
            "actor_loss",
            "mean"
        ),

        critic_loss_mean=(
            "critic_loss",
            "mean"
        ),

        gradient_variance_mean=(
            "gradient_variance",
            "mean"
        ),

        gradient_variance_std=(
            "gradient_variance",
            "std"
        ),

        gradient_norm_mean=(
            "gradient_norm",
            "mean"
        ),

        gradient_norm_std=(
            "gradient_norm",
            "std"
        ),
    )
    .reset_index()
)


window_statistics["episode_start"] = (
    window_statistics["window"]
    * WINDOW_SIZE
)

window_statistics["episode_end"] = (
    window_statistics["episode_start"]
    + WINDOW_SIZE
    - 1
)


window_statistics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "window_statistics.csv"
    ),
    index=False
)


# ============================================================
# AVERAGE WINDOW CURVES ACROSS SEEDS
# ============================================================

window_across_seeds = (
    window_statistics
    .groupby(
        [
            "model",
            "episode_start"
        ]
    )
    .agg(

        utilization_mean=(
            "utilization_mean",
            "mean"
        ),

        utilization_std_across_seeds=(
            "utilization_mean",
            "std"
        ),

        optimality_gap_mean=(
            "optimality_gap_mean",
            "mean"
        ),

        optimality_gap_std_across_seeds=(
            "optimality_gap_mean",
            "std"
        ),

        actor_loss_mean=(
            "actor_loss_mean",
            "mean"
        ),

        critic_loss_mean=(
            "critic_loss_mean",
            "mean"
        ),

        gradient_variance_mean=(
            "gradient_variance_mean",
            "mean"
        ),

        gradient_variance_std_across_seeds=(
            "gradient_variance_mean",
            "std"
        ),

        gradient_norm_mean=(
            "gradient_norm_mean",
            "mean"
        ),

        gradient_norm_std_across_seeds=(
            "gradient_norm_mean",
            "std"
        ),
    )
    .reset_index()
)


window_across_seeds.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "window_comparison_across_seeds.csv"
    ),
    index=False
)


# ============================================================
# PLOT FUNCTION
# ============================================================

def plot_metric(
    df,
    mean_column,
    std_column,
    ylabel,
    title,
    filename
):

    plt.figure(figsize=(11, 6))

    for model in df["model"].unique():

        model_df = df[
            df["model"] == model
        ]

        x = model_df["episode_start"]
        y = model_df[mean_column]
        std = model_df[std_column]

        plt.plot(
            x,
            y,
            label=model
        )

        # Mean ± STD across seeds
        plt.fill_between(
            x,
            y - std,
            y + std,
            alpha=0.15
        )

    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        dpi=300
    )

    plt.show()


# ============================================================
# LEARNING CURVES
# ============================================================

plot_metric(
    window_across_seeds,
    "utilization_mean",
    "utilization_std_across_seeds",
    "Mean Utilization",
    "Training Utilization — Mean ± STD Across Seeds",
    "utilization_comparison.png"
)


plot_metric(
    window_across_seeds,
    "optimality_gap_mean",
    "optimality_gap_std_across_seeds",
    "Mean Optimality Gap",
    "Training Optimality Gap — Mean ± STD Across Seeds",
    "optimality_gap_comparison.png"
)


plot_metric(
    window_across_seeds,
    "actor_loss_mean",
    "actor_loss_mean",
    "Mean Actor Loss",
    "Actor Loss During Training",
    "actor_loss_comparison.png"
)


plot_metric(
    window_across_seeds,
    "critic_loss_mean",
    "critic_loss_mean",
    "Mean Critic Loss",
    "Critic Loss During Training",
    "critic_loss_comparison.png"
)


plot_metric(
    window_across_seeds,
    "gradient_variance_mean",
    "gradient_variance_std_across_seeds",
    "Mean Gradient Variance",
    "Gradient Variance — Mean ± STD Across Seeds",
    "gradient_variance_comparison.png"
)


plot_metric(
    window_across_seeds,
    "gradient_norm_mean",
    "gradient_norm_std_across_seeds",
    "Mean Gradient Norm",
    "Gradient Norm — Mean ± STD Across Seeds",
    "gradient_norm_comparison.png"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 100)
print("FINAL MODEL COMPARISON")
print("=" * 100)

for _, row in model_statistics.iterrows():

    print(
        f"\n{row['model']}"
    )

    print(
        f"Utilization: "
        f"{row['utilization_mean']:.6f} "
        f"+/- "
        f"{row['utilization_seed_std']:.6f}"
    )

    print(
        f"Optimality Gap: "
        f"{row['optimality_gap_mean']:.6f} "
        f"+/- "
        f"{row['optimality_gap_seed_std']:.6f}"
    )

    print(
        f"Actor Loss: "
        f"{row['actor_loss_mean']:.6f} "
        f"+/- "
        f"{row['actor_loss_seed_std']:.6f}"
    )

    print(
        f"Critic Loss: "
        f"{row['critic_loss_mean']:.6f} "
        f"+/- "
        f"{row['critic_loss_seed_std']:.6f}"
    )

    print(
        f"Gradient Variance: "
        f"{row['gradient_variance_mean']:.6f} "
        f"+/- "
        f"{row['gradient_variance_seed_std']:.6f}"
    )

    print(
        f"Gradient Norm: "
        f"{row['gradient_norm_mean']:.6f} "
        f"+/- "
        f"{row['gradient_norm_seed_std']:.6f}"
    )


print("\n")
print("=" * 100)
print("DONE")
print("=" * 100)

print(
    f"All results saved in: {OUTPUT_DIR}"
)