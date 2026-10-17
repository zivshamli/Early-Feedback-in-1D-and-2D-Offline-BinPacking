
import torch
import numpy as np

from Actor2D import Actor, select_action
from Critic2D import Critic


class MonteCarloActorCritic2D:
    """
    Monte Carlo Actor-Critic for Offline 2D Bin Packing.

    Uses terminal reward only:

        R_terminal = -NumBins
                      + alpha * VolumeUtilization
                      + beta * PackingDensity
                      + mu * ItemStability

    Intermediate rewards returned by the environment are ignored.
    """

    def __init__(
        self,
        actor_state_dim,
        critic_state_dim,
        alpha=100.0,
        beta=100.0,
        mu=100.0,
        gamma=1.0,
        lr_actor=1e-4,
        lr_critic=1e-3,
        device=None,
        gradient_chunk_size=20,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # The 2D Actor adds ACTION_FEATURE_DIM internally.
        self.actor = Actor(actor_state_dim).to(self.device)
        self.critic = Critic(critic_state_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic
        )

        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.gamma = gamma
        self.gradient_chunk_size = gradient_chunk_size

    def calculate_terminal_reward(self, env):
        """Calculate the terminal reward from the final environment metrics."""
        metrics = env.calculate_metrics()

        num_bins = metrics["bins_used"]
        utilization = metrics["utilization"]
        packing_density = metrics["packing_density"]
        stability = metrics["stability"]

        reward = (
            -num_bins
            + self.alpha * utilization
            + self.beta * packing_density
            + self.mu * stability
        )

        return float(reward), metrics

    def generate_episode(self, env, episode=0):
        """
        Generate one complete episode.

        The environment may return intermediate rewards, but this
        Monte Carlo implementation uses only the final terminal reward.
        """
        state = env.reset()

        log_probs = []
        states = []
        done = False

        while not done:
            action, log_prob, state_tensor = select_action(
                self.actor,
                env,
                self.device,
                episode,
            )

            states.append(state_tensor)
            log_probs.append(log_prob)

            next_state, _, done, info = env.step(action)

            if info.get("invalid_action", False):
                raise RuntimeError(
                    f"Invalid action encountered during episode: {action}"
                )

            state = next_state

        terminal_reward, metrics = self.calculate_terminal_reward(env)

        return {
            "states": states,
            "log_probs": log_probs,
            "terminal_reward": terminal_reward,
            "metrics": metrics,
            "num_steps": len(log_probs),
        }

    def update(self, episode_data):
        """Update Actor and Critic using the terminal Monte Carlo return."""
        log_probs = episode_data["log_probs"]
        states = episode_data["states"]
        terminal_reward = episode_data["terminal_reward"]

        num_steps = len(log_probs)

        if num_steps == 0:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "gradient_variance": 0.0,
                "gradient_norm": 0.0,
            }

        # Return-to-go: only the terminal reward is non-zero.
        returns = [
            terminal_reward * (self.gamma ** (num_steps - 1 - t))
            for t in range(num_steps)
        ]

        returns = torch.tensor(
            returns, dtype=torch.float32, device=self.device
        )

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        actor_loss_value = 0.0
        critic_loss_value = 0.0
        gradient_norms = []

        chunk_size = max(1, self.gradient_chunk_size)

        for start in range(0, num_steps, chunk_size):
            end = min(start + chunk_size, num_steps)

            chunk_actor_loss = 0.0
            chunk_critic_loss = 0.0

            for t in range(start, end):
                state_tensor = states[t].to(self.device)
                value = self.critic(state_tensor).squeeze()

                target = returns[t]
                advantage = target - value

                actor_loss = -log_probs[t] * advantage.detach()
                critic_loss = advantage.pow(2)

                chunk_actor_loss = chunk_actor_loss + actor_loss
                chunk_critic_loss = chunk_critic_loss + critic_loss

                actor_loss_value += actor_loss.detach().item()
                critic_loss_value += critic_loss.detach().item()

            # Normalize by the total number of episode steps so that
            # chunking does not change the overall loss scale.
            chunk_actor_loss = chunk_actor_loss / num_steps
            chunk_critic_loss = chunk_critic_loss / num_steps

            chunk_actor_loss.backward()
            chunk_critic_loss.backward()

            actor_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=float("inf")
            )
            gradient_norms.append(float(actor_norm))

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        actor_loss_value /= num_steps
        critic_loss_value /= num_steps

        gradient_variance = (
            float(np.var(gradient_norms))
            if len(gradient_norms) > 1
            else 0.0
        )

        mean_gradient_norm = (
            float(np.mean(gradient_norms))
            if gradient_norms
            else 0.0
        )

        return {
            "actor_loss": actor_loss_value,
            "critic_loss": critic_loss_value,
            "gradient_variance": gradient_variance,
            "gradient_norm": mean_gradient_norm,
        }

    def train_episode(self, env, episode=0):
        """Generate an episode, update the networks, and return metrics."""
        episode_data = self.generate_episode(env, episode)
        update_metrics = self.update(episode_data)

        metrics = episode_data["metrics"]

        return {
            "reward": episode_data["terminal_reward"],
            "bins": metrics["bins_used"],
            "utilization": metrics["utilization"],
            "packing_density": metrics["packing_density"],
            "stability": metrics["stability"],
            "actor_loss": update_metrics["actor_loss"],
            "critic_loss": update_metrics["critic_loss"],
            "gradient_variance": update_metrics["gradient_variance"],
            "gradient_norm": update_metrics["gradient_norm"],
        }
