import torch
import torch.nn as nn

from Actor import Actor, select_action
from Critic import Critic


class TDEarlyFeedbackActorCritic:

    def __init__(
        self,
        actor_state_dim,
        critic_state_dim,
        alpha=100,
        gamma=0.99,
        feedback_gamma=1.0,
        lr_actor=1e-4,
        lr_critic=1e-3,
        device=None
    ):

        self.gradient_history = []
        self.gradient_variance_window = 100

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
        self.feedback_gamma = feedback_gamma

        # Actor gets state + item size + remaining capacity
        self.actor = Actor(
            actor_state_dim + 2
        ).to(self.device)

        print(
            "Actor input:",
            self.actor.network[0].in_features
        )

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

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=lr_actor
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=lr_critic
        )

    # =========================================================
    # Early Feedback Reward
    # =========================================================

    def calculate_early_reward(
        self,
        num_bins_before,
        num_bins_after,
        utilization_before,
        utilization_after
    ):

        delta_bins = (
            num_bins_after - num_bins_before
        )

        delta_utilization = (
            utilization_after
            - utilization_before
        )

        reward = (
            -delta_bins
            + self.feedback_gamma
            * self.alpha
            * delta_utilization
        )

        return float(reward)

    # =========================================================
    # Generate Episode
    # =========================================================

    def generate_episode(
        self,
        env,
        episode
    ):

        env.reset()

        trajectory = []

        done = False

        episode_reward = 0.0

        while not done:

            # -------------------------------------------------
            # State before action
            # -------------------------------------------------

            num_bins_before = len(env.bins)

            utilization_before = float(
                env._calculate_utilization()
            )

            # -------------------------------------------------
            # Select action
            # -------------------------------------------------

            action, log_prob, state_tensor = select_action(
                self.actor,
                env,
                self.device,
                episode
            )

            # -------------------------------------------------
            # Environment step
            # -------------------------------------------------

            next_state, _, done, info = env.step(
                action
            )

            # -------------------------------------------------
            # State after action
            # -------------------------------------------------

            num_bins_after = len(env.bins)

            utilization_after = float(
                env._calculate_utilization()
            )

            # -------------------------------------------------
            # Early Feedback
            # -------------------------------------------------

            reward = self.calculate_early_reward(
                num_bins_before,
                num_bins_after,
                utilization_before,
                utilization_after
            )

            episode_reward += reward

            # -------------------------------------------------
            # Next state
            # -------------------------------------------------
            next_state_vector = env.get_state_vector()

            next_state_tensor = torch.as_tensor(
                next_state_vector,
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)

            # -------------------------------------------------
            # Store transition
            #
            # IMPORTANT:
            # We detach next_state because the next state
            # does not belong to the actor computation graph.
            # -------------------------------------------------

            trajectory.append(
                {
                    "state": state_tensor,
                    "log_prob": log_prob,
                    "next_state": next_state_tensor,
                    "reward": reward,
                    "done": done
                }
            )

        return trajectory, episode_reward

    # =========================================================
    # TD Update
    # =========================================================

    def update(
        self,
        trajectory
    ):

        actor_losses = []
        critic_losses = []

        td_errors = []



        # -----------------------------------------------------
        # First calculate TD quantities
        # -----------------------------------------------------

        for step in trajectory:

            state = step["state"]
            next_state = step["next_state"]

            reward = float(step["reward"])
            done = step["done"]

            log_prob = step["log_prob"]

            # -------------------------------------------------
            # Critic value V(s)
            # -------------------------------------------------

            value = self.critic(state)

            # -------------------------------------------------
            # TD target
            # -------------------------------------------------

            with torch.no_grad():

                if done:

                    next_value = torch.zeros(
                        1,
                        device=self.device
                    )

                else:

                    next_value = self.critic(
                        next_state
                    )

                td_target = (
                    reward
                    + self.gamma
                    * next_value
                )

            # -------------------------------------------------
            # TD error
            # -------------------------------------------------

            td_error = (
                td_target - value
            )

            # -------------------------------------------------
            # Actor loss
            # -------------------------------------------------

            actor_loss = (
                -log_prob
                * td_error.detach()
            )

            # -------------------------------------------------
            # Critic loss
            # -------------------------------------------------

            critic_loss = nn.functional.mse_loss(
                value.squeeze(),
                td_target.squeeze()
            )

            actor_losses.append(
                actor_loss
            )

            critic_losses.append(
                critic_loss
            )

            td_errors.append(
                td_error.detach()
            )

        # =====================================================
        # Mean losses
        # =====================================================

        actor_loss = torch.stack(
            actor_losses
        ).mean()

        critic_loss = torch.stack(
            critic_losses
        ).mean()

        # =====================================================
        # Actor update
        # =====================================================

        self.actor_optimizer.zero_grad(
            set_to_none=True
        )

        actor_loss.backward()

        # =====================================================
        # Gradient statistics
        # =====================================================

        grads = []

        for parameter in self.actor.parameters():

            if parameter.grad is not None:

                grads.append(
                    parameter.grad.detach().flatten()
                )

        if len(grads) > 0:

            current_gradient = torch.cat(grads)

            # Gradient norm of current update
            gradient_norm = current_gradient.norm().item()

            # Store gradient on CPU
            self.gradient_history.append(
                current_gradient.detach().cpu()
            )

            # Keep only the latest 100 updates
            if len(self.gradient_history) > self.gradient_variance_window:

                self.gradient_history.pop(0)

            # Gradient variance across training updates
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

            else:

                gradient_variance = 0.0

        else:

            gradient_variance = 0.0
            gradient_norm = 0.0



        self.actor_optimizer.step()

        # =====================================================
        # Critic update
        # =====================================================

        self.critic_optimizer.zero_grad(
            set_to_none=True
        )

        critic_loss.backward()

        self.critic_optimizer.step()

        # =====================================================
        # Convert to Python numbers
        # =====================================================

        actor_loss_value = (
            actor_loss.detach().item()
        )

        critic_loss_value = (
            critic_loss.detach().item()
        )

        td_error_value = torch.stack(
            td_errors
        ).mean().item()



        # =====================================================
        # Free GPU memory
        # =====================================================

        del trajectory
        del actor_loss
        del critic_loss

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # =====================================================
        # Same return interface as before
        # =====================================================

        return (
            actor_loss_value,
            critic_loss_value,
            gradient_variance,
            gradient_norm
        )
