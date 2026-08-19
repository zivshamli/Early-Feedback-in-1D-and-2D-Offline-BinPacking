import torch
import torch.nn as nn
import numpy as np

from Actor import Actor, select_action
from Critic import Critic


class TDActorCritic:

    def __init__(
        self,
        actor_state_dim,
        critic_state_dim,
        alpha=100,
        gamma=0.99,
        lr_actor=1e-4,
        lr_critic=1e-3,
        device=None
    ):

        # =================================================
        # Device
        # =================================================

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        print("Using device:", self.device)

        self.alpha = alpha
        self.gamma = gamma

        # =================================================
        # Gradient statistics
        # =================================================

        self.gradient_history = []
        self.gradient_variance_window = 100

        # =================================================
        # Actor
        # =================================================

        self.actor = Actor(
            actor_state_dim + 2
        ).to(self.device)

        print(
            "Actor input:",
            self.actor.network[0].in_features
        )

        # =================================================
        # Critic
        # =================================================

        self.critic = Critic(
            critic_state_dim
        ).to(self.device)

        print(
            "Actor device:",
            next(self.actor.parameters()).device
        )

        print(
            "Critic device:",
            next(self.critic.parameters()).device
        )

        # =================================================
        # Optimizers
        # =================================================

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=lr_actor
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=lr_critic
        )

    # =========================================================
    # Terminal Reward
    # =========================================================

    def calculate_terminal_reward(self, env):

        num_bins = len(env.bins)

        utilization = float(
            env._calculate_utilization()
        )

        reward = (
            -num_bins
            + self.alpha * utilization
        )

        return float(reward)

    # =========================================================
    # Generate Episode
    #
    # IMPORTANT:
    #
    # We DO NOT store log_prob.
    #
    # log_prob contains a computational graph connected to
    # the Actor and would keep GPU memory alive for the whole
    # episode.
    #
    # Instead, we store:
    #
    #   - actor features for all valid actions
    #   - selected action index
    #
    # on CPU.
    #
    # During update(), log_prob is reconstructed.
    # =========================================================

    def generate_episode(
        self,
        env,
        episode
    ):

        env.reset()

        trajectory = []

        done = False

        while not done:

            # =================================================
            # Get valid actions BEFORE select_action
            # =================================================

            valid_actions = env.get_valid_actions()

            # =================================================
            # Build exactly the same features as select_action
            #
            # This duplicates the feature construction from
            # Actor.py so that we can reconstruct log_prob later.
            # =================================================

            state_vector = env.get_state_vector()

            features = []

            for action in valid_actions:

                item_index, bin_index = action

                # ---------------------------------------------
                # Item size
                # ---------------------------------------------

                item_size = env.remaining_items[item_index]

                # ---------------------------------------------
                # Remaining bin capacity
                # ---------------------------------------------

                if bin_index < len(env.bins):

                    used = sum(
                        env.bins[bin_index]
                    )

                    remaining_capacity = (
                        env.bin_capacity
                        - used
                    )

                else:

                    remaining_capacity = (
                        env.bin_capacity
                    )

                # ---------------------------------------------
                # State + action features
                # ---------------------------------------------

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

            # =================================================
            # Select action
            #
            # select_action still works exactly as before.
            # We simply DO NOT keep its log_prob.
            # =================================================

            action, log_prob, state_tensor = select_action(
                self.actor,
                env,
                self.device,
                episode
            )

            # =================================================
            # Find selected action index
            # =================================================

            action_index = valid_actions.index(action)

            # =================================================
            # Store Actor features on CPU
            #
            # This is detached from the Actor graph.
            #
            # Therefore the GPU computational graph created
            # inside select_action can be released immediately.
            # =================================================

            features_cpu = torch.tensor(
                np.asarray(
                    features,
                    dtype=np.float32
                ),
                dtype=torch.float32
            )

            # =================================================
            # Temperature used by select_action
            #
            # We store it so update() reproduces the exact
            # same policy calculation.
            # =================================================

            temperature = max(
                0.1,
                1.0 * (0.995 ** episode)
            )

            # =================================================
            # Environment step
            # =================================================

            _, _, done, info = env.step(
                action
            )

            # =================================================
            # Next state
            # =================================================

            next_state_vector = env.get_state_vector()

            next_state_tensor = torch.as_tensor(
                next_state_vector,
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)

            # =================================================
            # No Early Feedback
            # =================================================

            reward = 0.0

            trajectory.append(
                {
                    "state": state_tensor.detach(),
                    "features": features_cpu,
                    "action_index": action_index,
                    "temperature": temperature,
                    "next_state": next_state_tensor.detach(),
                    "reward": reward,
                    "done": done
                }
            )

            # =================================================
            # IMPORTANT:
            #
            # Do not keep select_action's computational graph.
            # =================================================

            del log_prob
            del features
            del valid_actions
            del state_vector

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # =====================================================
        # Terminal reward
        # =====================================================

        terminal_reward = self.calculate_terminal_reward(
            env
        )

        # Only final transition receives terminal reward

        trajectory[-1]["reward"] = terminal_reward

        episode_reward = terminal_reward

        return trajectory, episode_reward

    # =========================================================
    # TD(0) Update
    #
    # IMPORTANT:
    #
    # - Actor step ONLY ONCE at end
    # - Critic step ONLY ONCE at end
    # - No Early Feedback
    # - No Actor graphs stored in trajectory
    # =========================================================

    def update(
        self,
        trajectory
    ):

        num_steps = len(trajectory)

        if num_steps == 0:

            return (
                0.0,
                0.0,
                0.0,
                0.0
            )

        # =====================================================
        # Zero gradients ONCE
        # =====================================================

        self.actor_optimizer.zero_grad(
            set_to_none=True
        )

        self.critic_optimizer.zero_grad(
            set_to_none=True
        )

        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        td_error_sum = 0.0

        # =====================================================
        # Process transitions one at a time
        # =====================================================

        for step in trajectory:

            state = step["state"]
            next_state = step["next_state"]

            reward = float(
                step["reward"]
            )

            done = step["done"]

            features_cpu = step["features"]

            action_index = step["action_index"]

            temperature = step["temperature"]

            # =================================================
            # Move ONLY current transition's Actor features
            # to GPU.
            # =================================================

            features = features_cpu.to(
                self.device,
                non_blocking=True
            )

            # =================================================
            # Recalculate Actor probabilities
            #
            # This reproduces select_action().
            # =================================================

            action_scores = self.actor(
                features
            ).squeeze(-1)

            probabilities = torch.softmax(
                action_scores / temperature,
                dim=0
            )

            # =================================================
            # Recalculate selected action log probability
            # =================================================

            selected_probability = probabilities[
                action_index
            ]

            log_prob = torch.log(
                selected_probability
            )

            # =================================================
            # Critic V(s_t)
            # =================================================

            value = self.critic(
                state
            )

            # =================================================
            # TD Target
            # =================================================

            with torch.no_grad():

                if done:

                    next_value = torch.zeros(
                        1,
                        device=self.device,
                        dtype=torch.float32
                    )

                else:

                    next_value = self.critic(
                        next_state
                    )

                td_target = (
                    reward
                    + self.gamma * next_value
                )

            # =================================================
            # TD Error
            # =================================================

            td_error = (
                td_target - value
            )

            # =================================================
            # Actor Loss
            #
            # Same formula as original:
            #
            # -log_prob * TD_error.detach()
            #
            # Divide by num_steps because original code
            # averaged all transition losses.
            # =================================================

            actor_loss = (
                -log_prob
                * td_error.detach()
            )

            actor_loss = (
                actor_loss.mean()
                / num_steps
            )

            # =================================================
            # Critic Loss
            #
            # Same MSE formulation as original.
            # =================================================

            critic_loss = nn.functional.mse_loss(
                value.squeeze(),
                td_target.squeeze()
            )

            critic_loss = (
                critic_loss
                / num_steps
            )

            # =================================================
            # Actor backward
            #
            # Current transition's graph is released after
            # backward.
            #
            # Actor parameters are NOT changed yet.
            # =================================================

            actor_loss.backward()

            # =================================================
            # Critic backward
            #
            # Critic parameters are NOT changed yet.
            # =================================================

            critic_loss.backward()

            # =================================================
            # Python statistics
            # =================================================

            actor_loss_sum += (
                actor_loss.detach().item()
                * num_steps
            )

            critic_loss_sum += (
                critic_loss.detach().item()
                * num_steps
            )

            td_error_sum += (
                td_error.detach().mean().item()
            )

            # =================================================
            # Delete GPU tensors
            # =================================================

            del features
            del action_scores
            del probabilities
            del selected_probability
            del log_prob
            del value
            del next_value
            del td_target
            del td_error
            del actor_loss
            del critic_loss

        # =====================================================
        # Gradient Statistics
        #
        # Actor gradient is the accumulated gradient over
        # the whole episode.
        # =====================================================

        gradient_norm = 0.0
        gradient_variance = 0.0

        grads = []

        for parameter in self.actor.parameters():

            if parameter.grad is not None:

                grads.append(
                    parameter.grad.detach().flatten()
                )

        if len(grads) > 0:

            current_gradient = torch.cat(
                grads
            )

            # -------------------------------------------------
            # Gradient norm
            # -------------------------------------------------

            gradient_norm = (
                current_gradient.norm().item()
            )

            # -------------------------------------------------
            # Store gradient on CPU
            # -------------------------------------------------

            current_gradient_cpu = (
                current_gradient
                .detach()
                .cpu()
            )

            self.gradient_history.append(
                current_gradient_cpu
            )

            # -------------------------------------------------
            # Keep only latest 100
            # -------------------------------------------------

            if (
                len(self.gradient_history)
                > self.gradient_variance_window
            ):

                self.gradient_history.pop(0)

            # -------------------------------------------------
            # Gradient variance
            # -------------------------------------------------

            if len(self.gradient_history) > 1:

                gradient_matrix = torch.stack(
                    self.gradient_history
                )

                gradient_variance = (
                    gradient_matrix
                    .var(
                        dim=0,
                        unbiased=False
                    )
                    .mean()
                    .item()
                )

            # -------------------------------------------------
            # Free temporary tensors
            # -------------------------------------------------

            del current_gradient
            del current_gradient_cpu

        # =====================================================
        # Optimizer updates
        #
        # EXACTLY ONE update for the whole episode.
        # =====================================================

        self.actor_optimizer.step()

        self.critic_optimizer.step()

        # =====================================================
        # Average statistics
        # =====================================================

        actor_loss_value = (
            actor_loss_sum / num_steps
        )

        critic_loss_value = (
            critic_loss_sum / num_steps
        )

        td_error_value = (
            td_error_sum / num_steps
        )

        # =====================================================
        # Free trajectory
        # =====================================================

        del trajectory

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # =====================================================
        # Return
        # =====================================================

        return (
            actor_loss_value,
            critic_loss_value,
            gradient_variance,
            gradient_norm
        )