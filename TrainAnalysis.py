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
    "TD WITHOUT EF": "TDResult/TD_train_history_seed_*.csv",
    "TD_EF": "TDWithEF/TD_EF_train_history_seed_*.csv",
}

WINDOW_SIZE = 200

# Only for Utilization and Optimality Gap smoothing
ROLLING_WINDOW = 100
MAX_EPISODE = 5000

# ============================================================
# CONVERGENCE CONFIGURATION
# ============================================================

# Percentage of the best rolling utilization used as threshold
CONVERGENCE_THRESHOLD = 0.95

# Number of consecutive episodes that must remain above
# the threshold to consider the model converged
CONVERGENCE_PATIENCE = 100

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
# ============================================================
# TRAINING PERFORMANCE
# UTILIZATION + OPTIMALITY GAP
# ONLY THESE TWO USE ROLLING SMOOTHING
# ============================================================
# ============================================================

performance_data = data[
    data["episode"] <= MAX_EPISODE
].copy()

performance_data = (
    performance_data
    .sort_values(
        ["model", "seed", "episode"]
    )
    .reset_index(drop=True)
)


# ============================================================
# ROLLING MEAN PER SEED
# ============================================================

performance_data["utilization_smooth"] = (
    performance_data
    .groupby(
        ["model", "seed"]
    )["utilization"]
    .transform(
        lambda x: x.rolling(
            ROLLING_WINDOW,
            min_periods=1
        ).mean()
    )
)


performance_data["optimality_gap_smooth"] = (
    performance_data
    .groupby(
        ["model", "seed"]
    )["optimality_gap"]
    .transform(
        lambda x: x.rolling(
            ROLLING_WINDOW,
            min_periods=1
        ).mean()
    )
)


# ============================================================
# MEAN ± STD ACROSS SEEDS
# ============================================================

performance_across_seeds = (
    performance_data
    .groupby(
        [
            "model",
            "episode"
        ]
    )
    .agg(

        utilization_mean=(
            "utilization_smooth",
            "mean"
        ),

        utilization_std=(
            "utilization_smooth",
            "std"
        ),

        optimality_gap_mean=(
            "optimality_gap_smooth",
            "mean"
        ),

        optimality_gap_std=(
            "optimality_gap_smooth",
            "std"
        ),
    )
    .reset_index()
)


# ============================================================
# SAVE PERFORMANCE DATA
# ============================================================

performance_across_seeds.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "training_performance_rolling_100.csv"
    ),
    index=False
)


# ============================================================
# ============================================================
# CONVERGENCE SPEED + AUC
# ============================================================
# ============================================================

def calculate_auc(episodes, utilization):
    """
    Calculate normalized AUC using the trapezoidal rule.

    The result represents the average utilization
    over the training process.
    """

    episodes = np.asarray(episodes)
    utilization = np.asarray(utilization)

    if len(episodes) < 2:
        return np.nan

    episode_range = (
        episodes[-1] - episodes[0]
    )

    if episode_range <= 0:
        return np.nan

    auc = np.trapezoid(
        utilization,
        episodes
    )

    normalized_auc = (
        auc / episode_range
    )

    return normalized_auc


def calculate_convergence_episode(
    episodes,
    utilization_smooth,
    threshold_ratio=0.95,
    patience=100
):
    """
    Calculate Episodes-to-Convergence.

    Convergence is defined as the first episode where
    rolling utilization reaches at least:

        threshold_ratio * maximum rolling utilization

    and remains above that threshold for 'patience'
    consecutive episodes.

    Returns NaN if convergence is not reached.
    """

    episodes = np.asarray(episodes)

    utilization_smooth = np.asarray(
        utilization_smooth
    )

    if len(episodes) == 0:
        return np.nan

    # --------------------------------------------------------
    # Best rolling utilization for this seed
    # --------------------------------------------------------

    max_utilization = np.nanmax(
        utilization_smooth
    )

    threshold = (
        threshold_ratio
        * max_utilization
    )

    # --------------------------------------------------------
    # Identify episodes above threshold
    # --------------------------------------------------------

    above_threshold = (
        utilization_smooth >= threshold
    )

    # --------------------------------------------------------
    # Find first run of 'patience' consecutive episodes
    # --------------------------------------------------------

    consecutive_count = 0

    for i, is_above in enumerate(
        above_threshold
    ):

        if is_above:

            consecutive_count += 1

            if consecutive_count >= patience:

                convergence_index = (
                    i - patience + 1
                )

                return episodes[
                    convergence_index
                ]

        else:

            consecutive_count = 0

    return np.nan


# ============================================================
# CALCULATE AUC + CONVERGENCE FOR EACH SEED
# ============================================================

convergence_results = []

for (model, seed), group in performance_data.groupby(
    ["model", "seed"]
):

    group = (
        group
        .sort_values("episode")
        .reset_index(drop=True)
    )

    episodes = group[
        "episode"
    ].to_numpy()

    utilization = group[
        "utilization"
    ].to_numpy()

    utilization_smooth = group[
        "utilization_smooth"
    ].to_numpy()

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    auc = calculate_auc(
        episodes,
        utilization
    )

    # --------------------------------------------------------
    # Maximum rolling utilization
    # --------------------------------------------------------

    max_rolling_utilization = np.nanmax(
        utilization_smooth
    )

    # --------------------------------------------------------
    # Convergence threshold
    # --------------------------------------------------------

    convergence_threshold = (
        CONVERGENCE_THRESHOLD
        * max_rolling_utilization
    )

    # --------------------------------------------------------
    # Episodes-to-Convergence
    # --------------------------------------------------------

    convergence_episode = (
        calculate_convergence_episode(
            episodes,
            utilization_smooth,
            threshold_ratio=CONVERGENCE_THRESHOLD,
            patience=CONVERGENCE_PATIENCE
        )
    )

    convergence_results.append({

        "model": model,

        "seed": seed,

        "AUC": auc,

        "max_rolling_utilization":
            max_rolling_utilization,

        "convergence_threshold":
            convergence_threshold,

        "episodes_to_convergence":
            convergence_episode,
    })


# ============================================================
# PER-SEED CONVERGENCE RESULTS
# ============================================================

convergence_per_seed = pd.DataFrame(
    convergence_results
)


convergence_per_seed = (
    convergence_per_seed
    .sort_values(
        ["model", "seed"]
    )
    .reset_index(drop=True)
)


print("\n")
print("=" * 100)
print("CONVERGENCE SPEED + AUC — PER SEED")
print("=" * 100)

print(
    convergence_per_seed.to_string(
        index=False
    )
)


convergence_per_seed.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "convergence_speed_per_seed.csv"
    ),
    index=False
)


# ============================================================
# SUMMARY ACROSS SEEDS
# ============================================================

convergence_summary = (
    convergence_per_seed
    .groupby("model")
    .agg(

        AUC_mean=(
            "AUC",
            "mean"
        ),

        AUC_std=(
            "AUC",
            "std"
        ),

        convergence_episodes_mean=(
            "episodes_to_convergence",
            "mean"
        ),

        convergence_episodes_std=(
            "episodes_to_convergence",
            "std"
        ),

        max_utilization_mean=(
            "max_rolling_utilization",
            "mean"
        ),

        max_utilization_std=(
            "max_rolling_utilization",
            "std"
        ),
    )
    .reset_index()
)


print("\n")
print("=" * 100)
print("CONVERGENCE SPEED + AUC — MEAN ± STD ACROSS SEEDS")
print("=" * 100)

print(
    convergence_summary.to_string(
        index=False
    )
)


convergence_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "convergence_speed_summary.csv"
    ),
    index=False
)


# ============================================================
# PERFORMANCE PLOT FUNCTION
# ============================================================

def plot_training_performance(
    df,
    mean_column,
    std_column,
    ylabel,
    title,
    filename
):

    plt.figure(figsize=(11, 6))

    for model in df["model"].unique():

        model_df = (
            df[
                df["model"] == model
            ]
            .sort_values("episode")
        )

        x = model_df["episode"]

        y = model_df[mean_column]

        std = (
            model_df[std_column]
            .fillna(0)
        )

        plt.plot(
            x,
            y,
            label=model
        )

        plt.fill_between(
            x,
            y - std,
            y + std,
            alpha=0.15
        )

    plt.xlabel("Episode")
    plt.ylabel(ylabel)

    plt.title(title)

    # Force the graph to show up to Episode 5000
    plt.xlim(
        0,
        MAX_EPISODE
    )

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
# UTILIZATION — ROLLING 100
# ============================================================

plot_training_performance(
    performance_across_seeds,
    "utilization_mean",
    "utilization_std",
    "Mean Utilization",
    "Training Utilization — Mean ± STD Across Seeds with Rolling Window (100 Episodes)",
    "utilization_comparison.png"
)


# ============================================================
# OPTIMALITY GAP — ROLLING 100
# ============================================================

plot_training_performance(
    performance_across_seeds,
    "optimality_gap_mean",
    "optimality_gap_std",
    "Mean Optimality Gap",
    "Training Optimality Gap — Mean ± STD Across Seeds with Rolling Window (100 Episodes)",
    "optimality_gap_comparison.png"
)

# ============================================================
# CONVERGENCE SPEED + AUC VISUALIZATION
# ============================================================

def plot_convergence_speed(
    performance_df,
    convergence_df,
    filename
):
    """
    Visualize convergence speed and AUC during training.

    The plot shows:
    - Rolling utilization averaged across seeds
    - +/- 1 STD across seeds
    - Mean Episodes-to-Convergence as a vertical dashed line
    - AUC value for each model
    """

    plt.figure(figsize=(12, 7))

    models = performance_df["model"].unique()

    for model in models:

        model_df = (
            performance_df[
                performance_df["model"] == model
            ]
            .sort_values("episode")
        )

        x = model_df["episode"]

        y = model_df["utilization_mean"]

        std = (
            model_df["utilization_std"]
            .fillna(0)
        )

        # ----------------------------------------------------
        # Mean rolling utilization
        # ----------------------------------------------------

        plt.plot(
            x,
            y,
            label=model
        )

        # ----------------------------------------------------
        # +/- 1 STD across seeds
        # ----------------------------------------------------

        plt.fill_between(
            x,
            y - std,
            y + std,
            alpha=0.15
        )

        # ----------------------------------------------------
        # Get convergence + AUC information
        # ----------------------------------------------------

        model_convergence = convergence_df[
            convergence_df["model"] == model
        ]

        if len(model_convergence) == 0:
            continue

        mean_convergence = (
            model_convergence[
                "convergence_episodes_mean"
            ]
            .iloc[0]
        )

        mean_auc = (
            model_convergence[
                "AUC_mean"
            ]
            .iloc[0]
        )

        # ----------------------------------------------------
        # Vertical convergence line
        # ----------------------------------------------------

        if pd.notna(mean_convergence):

            plt.axvline(
                mean_convergence,
                linestyle="--",
                alpha=0.8
            )

            # ------------------------------------------------
            # Find utilization near convergence
            # ------------------------------------------------

            closest_idx = (
                (
                    model_df["episode"]
                    - mean_convergence
                )
                .abs()
                .idxmin()
            )

            convergence_y = (
                model_df.loc[
                    closest_idx,
                    "utilization_mean"
                ]
            )

            # ------------------------------------------------
            # Convergence label
            # ------------------------------------------------

            plt.annotate(
                f"Convergence ≈ "
                f"{mean_convergence:.0f}",
                xy=(
                    mean_convergence,
                    convergence_y
                ),
                xytext=(
                    8,
                    15
                ),
                textcoords="offset points",
                fontsize=9
            )

    # ========================================================
    # AUC TEXT BOX
    # ========================================================

    auc_lines = []

    for model in models:

        model_convergence = convergence_df[
            convergence_df["model"] == model
        ]

        if len(model_convergence) == 0:
            continue

        mean_auc = (
            model_convergence[
                "AUC_mean"
            ]
            .iloc[0]
        )

        std_auc = (
            model_convergence[
                "AUC_std"
            ]
            .iloc[0]
        )

        if pd.notna(mean_auc):

            auc_lines.append(
                f"{model}: "
                f"AUC = {mean_auc:.4f} "
                f"± {std_auc:.4f}"
            )

    auc_text = (
        "Learning Efficiency (AUC)\n"
        + "\n".join(auc_lines)
    )

    # --------------------------------------------------------
    # Add AUC information to the graph
    # --------------------------------------------------------

    plt.text(
        0.02,
        0.03,
        auc_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round",
            alpha=0.85
        )
    )

    # ========================================================
    # GRAPH FORMATTING
    # ========================================================

    plt.xlabel("Episode")

    plt.ylabel("Mean Utilization")

    plt.title(
        "Convergence Speed and Learning Efficiency"
    )

    # Force graph to show up to Episode 5000
    plt.xlim(
        0,
        MAX_EPISODE
    )

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
# CREATE CONVERGENCE SPEED + AUC PLOT
# ============================================================

plot_convergence_speed(
    performance_across_seeds,
    convergence_summary,
    "convergence_speed_comparison.png"
)

# ============================================================
# ============================================================
# WINDOW ANALYSIS
# EVERYTHING BELOW REMAINS AS IN YOUR ORIGINAL CODE
# ============================================================
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

# NOTE:
# Utilization and Optimality Gap are NOT plotted here anymore.
# They are plotted above using Rolling Mean.
#
# The four plots below remain unchanged.


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


# ============================================================
# FINAL CONVERGENCE SUMMARY
# ============================================================

print("\n")
print("=" * 100)
print("FINAL CONVERGENCE SPEED + AUC")
print("=" * 100)

for _, row in convergence_summary.iterrows():

    print(
        f"\n{row['model']}"
    )

    print(
        f"AUC: "
        f"{row['AUC_mean']:.6f} "
        f"+/- "
        f"{row['AUC_std']:.6f}"
    )

    if pd.notna(
        row["convergence_episodes_mean"]
    ):

        print(
            f"Convergence Episodes: "
            f"{row['convergence_episodes_mean']:.2f} "
            f"+/- "
            f"{row['convergence_episodes_std']:.2f}"
        )

    else:

        print(
            "Convergence Episodes: "
            "Not reached"
        )

    print(
        f"Maximum Rolling Utilization: "
        f"{row['max_utilization_mean']:.6f} "
        f"+/- "
        f"{row['max_utilization_std']:.6f}"
    )


print("\n")
print("=" * 100)
print("DONE")
print("=" * 100)

print(
    f"All results saved in: {OUTPUT_DIR}"
)