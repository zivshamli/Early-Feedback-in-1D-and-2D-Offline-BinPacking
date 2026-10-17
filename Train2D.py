
import random
import numpy as np
import torch
import pandas as pd
import gc

from Packlib2DLoader import PackLib2DLoader
from TwoDBinPackingEnv import Offline2DBinPackingEnv
from MCAgent2D import MonteCarloActorCritic2D


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ==================================================
# Evaluation
# ==================================================

def evaluate(agent, instances):

    results = []

    with torch.no_grad():

        for instance in instances:

            env = Offline2DBinPackingEnv(
                instance.items,
                instance.bin_width,
                instance.bin_height,
                max_items=200,
                stability_threshold=0.5,
                grid_size=1
            )

            _, reward = agent.generate_episode(
                env,
                episode=0
            )

            metrics = env.calculate_metrics()

            bins = len(env.bins)

            results.append({

                "instance": instance.name,

                "reward": reward,

                "bins": bins,

                "optimal_bins": instance.optimal_bins,

                "utilization": metrics["utilization"],

                "packing_density": metrics["packing_density"],

                "stability": metrics["stability"]

            })

            del env

            gc.collect()

    return results


# ==================================================
# Validation metrics
# ==================================================

def validate(agent, validation_instances):

    results = evaluate(
        agent,
        validation_instances
    )

    utilizations = []

    gaps = []

    for r in results:

        utilizations.append(
            r["utilization"]
        )

        gap = (
            r["bins"] - r["optimal_bins"]
        ) / r["optimal_bins"]

        gaps.append(gap)

    return {

        "utilization": np.mean(utilizations),

        "optimality_gap": np.mean(gaps)

    }


# ==================================================
# Training
# ==================================================

def train(
    agent,
    train_instances,
    validation_instances,
    episodes,
    validation_interval=200
):

    history = []

    validation_history = []

    for episode in range(episodes):

        # =================================
        # Select random training instance
        # =================================

        instance = random.choice(train_instances)

        env = Offline2DBinPackingEnv(
            instance.items,
            instance.bin_width,
            instance.bin_height,
            max_items=200,
            stability_threshold=0.5,
            grid_size=1
        )

        # =================================
        # Generate episode
        # =================================

        episode_data = agent.generate_episode(
            env,
            episode
        )

        update_metrics = agent.update(episode_data)

        # Terminal reward and final environment metrics
        reward = episode_data["terminal_reward"]
        metrics = episode_data["metrics"]

        # Extract training metrics
        actor_loss = update_metrics["actor_loss"]
        critic_loss = update_metrics["critic_loss"]
        gradient_variance = update_metrics["gradient_variance"]
        gradient_norm = update_metrics["gradient_norm"]

        # Extract episode metrics
        bins = metrics["bins_used"]
        utilization = metrics["utilization"]
        packing_density = metrics["packing_density"]
        stability = metrics["stability"]
        

        # =================================
        # Free memory
        # =================================

        
        del env

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # =================================
        # Validation
        # =================================

        if (
            episode % validation_interval == 0
            and episode != 0
        ):

            print("\nRunning Validation...")

            val = validate(
                agent,
                validation_instances
            )

            validation_history.append({

                "episode": episode,

                "utilization": val["utilization"],

                "optimality_gap": val["optimality_gap"]

            })

            print(
                "Validation Episode:",
                episode
            )

            print(
                "Validation Utilization:",
                val["utilization"]
            )

            print(
                "Validation Optimality Gap:",
                val["optimality_gap"]
            )

        # =================================
        # Training log
        # =================================

        print(
            "Episode:",
            episode,
            "Instance:",
            instance.name,
            "Reward:",
            reward,
            "Bins:",
            bins,
            "OPT:",
            instance.optimal_bins,
            "Util:",
            utilization,
            "Density:",
            packing_density,
            "Stability:",
            stability
        )

    return history, validation_history


# ==================================================
# Load Dataset
# ==================================================

loader = PackLib2DLoader(
    "packlib2"
)

loader.load()


# ==================================================
# Split Dataset
# ==================================================

train_instances, validation_instances, test_instances = (
    loader.train_val_test_split(
        test_size=0.2,
        val_size=0.1,
        seed=42
    )
)


print("Train:", len(train_instances))
print("Validation:", len(validation_instances))
print("Test:", len(test_instances))


# ==================================================
# Multiple Seeds
# ==================================================

seeds = [1]

for seed in seeds:

    all_results = []

    print("\n===================")
    print("Seed:", seed)
    print("===================")

    set_seed(seed)

    # =================================
    # Determine state dimension
    # =================================

    dummy_env = Offline2DBinPackingEnv(
        train_instances[0].items,
        train_instances[0].bin_width,
        train_instances[0].bin_height,
        max_items=200,
        stability_threshold=0.5,
        grid_size=1
    )


    state_dim = len(
        dummy_env.get_state_vector()
    )

    del dummy_env
    gc.collect()



    # =================================
    # Initialize Monte Carlo Agent
    # =================================

    agent = MonteCarloActorCritic2D(
        actor_state_dim=state_dim,
        critic_state_dim=state_dim,
        alpha=100.0,
        beta=10.0,
        mu=5.0
    )

    agent_type = "MC_2D"

    print("Agent is MonteCarloActorCritic")

    # =================================
    # Training
    # =================================

    train_history, validation_history = train(
        agent,
        train_instances,
        validation_instances,
        episodes=5000,
        validation_interval=200
    )

    # =================================
    # Test Evaluation
    # =================================

    test_results = evaluate(
        agent,
        test_instances
    )

    print("\nTest Results:")

    for result in test_results:

        print(
            f"Instance: {result['instance']}, "
            f"Reward: {result['reward']}, "
            f"Bins: {result['bins']}, "
            f"Optimal Bins: {result['optimal_bins']}, "
            f"Utilization: {result['utilization']}, "
            f"Packing Density: {result['packing_density']}, "
            f"Stability: {result['stability']}"
        )

    all_results.append({

        "seed": seed,

        "train_history": train_history,

        "validation_history": validation_history,

        "test_results": test_results

    })

    # =================================
    # Save Results
    # =================================

    pd.DataFrame(train_history).to_csv(
        f"{agent_type}_train_history_seed_{seed}.csv",
        index=False
    )

    pd.DataFrame(validation_history).to_csv(
        f"{agent_type}_validation_history_seed_{seed}.csv",
        index=False
    )

    pd.DataFrame(test_results).to_csv(
        f"{agent_type}_test_results_seed_{seed}.csv",
        index=False
    )

    # =================================
    # Cleanup
    # =================================

    del agent
    del train_history
    del validation_history
    del test_results

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()


print("\nFinished - Results Saved")
