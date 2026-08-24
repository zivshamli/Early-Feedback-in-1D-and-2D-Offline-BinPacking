import random
import numpy as np
import torch
import pandas as pd
import gc


from ORLibLoader import ORLib1DBinPackingLoader
from OneDBinPackingEnv import Offline1DBinPackingEnv
from MCAgent import MonteCarloActorCritic
from TDEarlyFeedBackAgent import TDEarlyFeedbackActorCritic
from TDAgent import TDActorCritic



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

            env = Offline1DBinPackingEnv(
                instance.items,
                instance.capacity
            )


            _, reward = agent.generate_episode(
                env,
                episode=0
            )


            results.append({

                "instance": instance.name,

                "reward": reward,

                "bins": len(env.bins),

                "optimal_bins": instance.optimal_bins,

                "utilization": env._calculate_utilization()

            })


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


        # לבחור instance אקראי לאימון
        instance = random.choice(train_instances)
    


        env = Offline1DBinPackingEnv(
            instance.items,
            instance.capacity
        )


        trajectory, reward = agent.generate_episode(
            env,
            episode
        )


        (
            actor_loss,
            critic_loss,
            gradient_variance,
            gradient_norm

        ) = agent.update(
            trajectory,
            reward
        )

        bins = len(env.bins)

        utilization = env._calculate_utilization()
        trajectory.clear()
        del trajectory
        del env
        gc.collect()
        torch.cuda.empty_cache()




        history.append({

            "episode": episode,

            "reward": reward,

            "bins": bins,

            "optimal_bins": instance.optimal_bins,

            "utilization": utilization,

            "actor_loss": actor_loss,

            "critic_loss": critic_loss,

            "gradient_variance": gradient_variance,

            "gradient_norm": gradient_norm

        })



        # =================================
        # Validation every 200 episodes
        # =================================

        if episode % validation_interval == 0 and episode != 0:


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



        if episode % 1 == 0:

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
                utilization
            )



    return history, validation_history


def td_ef_train(agent,
        train_instances,
        validation_instances,
        episodes,
        validation_interval=200
):

    history = []

    validation_history = []


    for episode in range(episodes):


        # לבחור instance אקראי לאימון
        instance = random.choice(train_instances)
    


        env = Offline1DBinPackingEnv(
            instance.items,
            instance.capacity
        )


        trajectory, reward = agent.generate_episode(
            env,
            episode
        )


        (
            actor_loss,
            critic_loss,
            gradient_variance,
            gradient_norm

        ) = agent.update(
            trajectory
            
        )

        bins = len(env.bins)

        utilization = env._calculate_utilization()
        episode_reward=sum([step["reward"] for step in trajectory])
        trajectory.clear()
        del trajectory
        del env
        gc.collect()
        torch.cuda.empty_cache()




        history.append({

            "episode": episode,

            "reward": episode_reward,

            "bins": bins,

            "optimal_bins": instance.optimal_bins,

            "utilization": utilization,

            "actor_loss": actor_loss,

            "critic_loss": critic_loss,

            "gradient_variance": gradient_variance,

            "gradient_norm": gradient_norm

        })



        # =================================
        # Validation every 200 episodes
        # =================================

        if episode % validation_interval == 0 and episode != 0:


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



        if episode % 1 == 0:

            print(                
                "Episode:",
                episode,
                "Instance:",
                instance.name,
                "Reward:",
                episode_reward,
                "Bins:",
                bins,
                "OPT:",
                instance.optimal_bins,
                "Util:",
                utilization
            )



    return history, validation_history

def td_train(
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

        env = Offline1DBinPackingEnv(
            instance.items,
            instance.capacity
        )

        # =================================
        # Generate episode
        # =================================

        trajectory, episode_reward = agent.generate_episode(
            env,
            episode
        )

        # =================================
        # TD(0) update
        # =================================

        (
            actor_loss,
            critic_loss,
            gradient_variance,
            gradient_norm
        ) = agent.update(
            trajectory
        )

        # =================================
        # Episode metrics
        # =================================

        bins = len(env.bins)

        utilization = env._calculate_utilization()

        # =================================
        # Save history
        # =================================

        history.append({

            "episode": episode,

            "reward": episode_reward,

            "bins": bins,

            "optimal_bins": instance.optimal_bins,

            "utilization": utilization,

            "actor_loss": actor_loss,

            "critic_loss": critic_loss,

            "gradient_variance": gradient_variance,

            "gradient_norm": gradient_norm

        })

        # =================================
        # Free memory
        # =================================

        trajectory.clear()

        del trajectory
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
            episode_reward,
            "Bins:",
            bins,
            "OPT:",
            instance.optimal_bins,
            "Util:",
            utilization
        )

    return history, validation_history


# ==================================================
# Load Dataset
# ==================================================

loader = ORLib1DBinPackingLoader(
    "ORLib"
)

loader.load()



# ==================================================
# Split Dataset
# ==================================================

train_instances, validation_instances, test_instances = loader.train_val_test_split(test_size=0.2,val_size=0.15,seed=42)



print("Train:", len(train_instances))
print("Validation:", len(validation_instances))
print("Test:", len(test_instances))



# ==================================================
# Multiple Seeds
# ==================================================

seeds = [3,5,8,10,11,12,13,14,15,16,18,20,42]

for seed in seeds:

    all_results = []
    print("\n===================")
    print("Seed:", seed)
    print("===================")


    set_seed(seed)



    dummy_env = Offline1DBinPackingEnv(
        train_instances[0].items,
        train_instances[0].capacity
    )


    state_dim = len(
        dummy_env.get_state_vector()
    )

    # dummy environment is no longer needed, delete it to free memory,only for declaration of state_dim, as the state dimension is the same for all instances in the dataset.
    del dummy_env
    gc.collect()

    # now we can initialize the agent with the correct state dimension,and train the agent on the training instances, validate on the validation instances, and finally evaluate on the test instances.

    '''
    agent = MonteCarloActorCritic(

        actor_state_dim=state_dim,

        critic_state_dim=state_dim

    )
    if agent.__class__.__name__ == "MonteCarloActorCritic":
        agent_type="MC"
        print("Agent is MonteCarloActorCritic")
    
    train_history, validation_history = train(

        agent,

        train_instances,

        validation_instances,

        episodes=5000,

        validation_interval=200

    )

    '''
    '''
    agent = TDEarlyFeedbackActorCritic(
        actor_state_dim=state_dim,

        critic_state_dim=state_dim

    )

    if agent.__class__.__name__ == "TDEarlyFeedbackActorCritic":
        agent_type="TD_EF"
        print("Agent is TDEarlyFeedbackActorCritic")

    train_history, validation_history = td_ef_train(
        agent,
        train_instances,
        validation_instances,
        episodes=5000,
        validation_interval=200
    )
    
    '''
    agent = TDActorCritic(
        actor_state_dim=state_dim,
        critic_state_dim=state_dim
    )
    if agent.__class__.__name__ == "TDActorCritic":
        agent_type="TD"
        print("Agent is TDActorCritic")
    train_history, validation_history = td_train(
        agent,
        train_instances,
        validation_instances,
        episodes=5000,
        validation_interval=200
    )

    

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
            f"Utilization: {result['utilization']}"
        )



    all_results.append({

        "seed": seed,

        "train_history": train_history,

        "validation_history": validation_history,

        "test_results": test_results

    })
# ==================================================
# Save Results
# ==================================================

    # Save train history
    pd.DataFrame(train_history).to_csv(
        f"{agent_type}_train_history_seed_{seed}.csv",
        index=False
    )

    # Save validation history
    pd.DataFrame(validation_history).to_csv(
        f"{agent_type}_validation_history_seed_{seed}.csv",
        index=False
    )

    # Save test results
    pd.DataFrame(test_results).to_csv(
        f"{agent_type}_test_results_seed_{seed}.csv",
        index=False
    )

    del agent
    del train_history
    del validation_history
    del test_results
    torch.cuda.empty_cache()
    gc.collect()
    

print("\nFinished - Results Saved")