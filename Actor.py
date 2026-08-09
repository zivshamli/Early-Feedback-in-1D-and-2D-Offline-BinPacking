import torch
import torch.nn as nn
import numpy as np



class Actor(nn.Module):

    def __init__(self, state_dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(state_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )


    def forward(self, state):

        return self.network(state)



def select_action(
        actor,
        env,
        device,
        episode
):


    valid_actions = env.get_valid_actions()

    state_vector = env.get_state_vector()



    features = []



    for action in valid_actions:


        item_index, bin_index = action


        # גודל הפריט
        item_size = env.remaining_items[item_index]



        # קיבולת פנויה בבין
        if bin_index < len(env.bins):

            used = sum(env.bins[bin_index])

            remaining_capacity = (
                env.bin_capacity - used
            )

        else:

            # פתיחת bin חדש
            remaining_capacity = env.bin_capacity



        # state + action features
        feature = np.concatenate(
            [
                np.array(
                    state_vector,
                    dtype=np.float32
                ),

                np.array(
                    [
                        item_size,
                        remaining_capacity
                    ],
                    dtype=np.float32
                )
            ]
        )


        features.append(feature)



    # Batch של כל הפעולות
    features = torch.tensor(
        np.array(features),
        dtype=torch.float32,
        device=device
    )



    # Forward אחד במקום אחד לכל פעולה
    action_scores = actor(
        features
    ).squeeze(-1)



    temperature = max(
        0.1,
        1.0 * (0.995 ** episode)
    )  # temperature parameter for softmax

    probabilities = torch.softmax(
        action_scores/temperature,
        dim=0
    )



    action_index = torch.multinomial(
        probabilities,
        1
    )



    selected_action = valid_actions[
        action_index.item()
    ]



    log_prob = torch.log(
        probabilities[
            action_index.item()
        ]
    )



    # state עבור Critic
    state_tensor = torch.tensor(
        state_vector,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)



    return (
        selected_action,
        log_prob,
        state_tensor
    )