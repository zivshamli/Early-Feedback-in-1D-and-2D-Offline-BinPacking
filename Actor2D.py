import torch
import torch.nn as nn
import numpy as np


ACTION_FEATURE_DIM = 9


class Actor(nn.Module):

    def __init__(self, state_dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(state_dim + ACTION_FEATURE_DIM, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, state_action):

        return self.network(state_action)


def select_action(
    actor,
    env,
    device,
    episode
):

    valid_actions = env.get_valid_actions()

    if not valid_actions:
        raise ValueError(
            "No valid actions available in the environment."
        )

    state_vector = env.get_state_vector()

    bin_area = env.bin_width * env.bin_height

    features = []

    for action in valid_actions:

        item_index, bin_index, x, y, rotation = action

        # Original item dimensions
        item_width, item_height = (
            env.remaining_items[item_index]
        )

        # Dimensions after rotation
        if rotation == 90:
            placed_width = item_height
            placed_height = item_width
        else:
            placed_width = item_width
            placed_height = item_height

        # Check whether a new bin is being opened
        is_new_bin = int(bin_index == len(env.bins))

        if not is_new_bin:

            bin_items = env.bins[bin_index]

            used_area = sum(
                item["width"] * item["height"]
                for item in bin_items
            )

        else:
            used_area = 0.0

        # Normalize selected action features
        normalized_x = x / env.bin_width
        normalized_y = y / env.bin_height

        normalized_used_area = (
            used_area / bin_area
            if bin_area > 0 else 0.0
        )

        # State + action features
        action_features = np.array(
            [
                item_width,
                item_height,
                placed_width,
                placed_height,
                normalized_x,
                normalized_y,
                normalized_used_area,
                rotation / 90.0,
                float(is_new_bin)
            ],
            dtype=np.float32
        )

        feature = np.concatenate(
            [
                np.asarray(
                    state_vector,
                    dtype=np.float32
                ),
                action_features
            ]
        )

        features.append(feature)

    # Batch of all valid actions
    features = torch.tensor(
        np.asarray(features),
        dtype=torch.float32,
        device=device
    )

    # Score all actions in one forward pass
    action_scores = actor(
        features
    ).squeeze(-1)

    # Exploration temperature
    temperature = max(
        0.1,
        1.0 * (0.995 ** episode)
    )

    probabilities = torch.softmax(
        action_scores / temperature,
        dim=0
    )

    # Sample an action
    action_index = torch.multinomial(
        probabilities,
        1
    ).item()

    selected_action = valid_actions[action_index]

    log_prob = torch.log(
        probabilities[action_index].clamp_min(1e-8)
    )

    # State for the Critic
    state_tensor = torch.as_tensor(
        state_vector,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    return (
        selected_action,
        log_prob,
        state_tensor
    )