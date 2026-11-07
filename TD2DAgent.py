
import torch
import numpy as np

from Actor2D import Actor, select_action
from Critic2D import Critic


class TDActorCritic2D:
    """
    TD(0) Actor-Critic for Offline 2D Bin Packing
    WITHOUT Early Feedback.

    =====================================================
    REWARD FUNCTION
    =====================================================

    Intermediate rewards:
        r_t = 0

    Terminal reward:

        R_terminal = -NumBins
                      + alpha * VolumeUtilization
                      + beta * PackingDensity
                      + mu * ItemStability

    =====================================================
    TD(0) TARGET
    =====================================================

    For intermediate transitions:

        target = 0 + gamma * V(next_state)

    For terminal transitions:

        target = R_terminal

    =====================================================
    IMPORTANT
    =====================================================

    - No intermediate Early Feedback rewards.
    - Bootstrapping is performed using the Critic.
    - Terminal objective is identical to the MC agent.
    - Actor and Critic use the numerical state vector.
    - Gradient statistics are calculated per episode.
    """

    def __init__(
        self,
        actor_state_dim,
        critic_state_dim,
        alpha=100.0,
        beta=1.0,
        mu=1.0,
        gamma=1.0,
        lr_actor=1e-4,
        lr_critic=1e-3,
        device=None,
        gradient_chunk_size=20,
    ):

        # =================================================
        # Device Configuration
        # =================================================

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # =================================================
        # Actor and Critic Networks
        # =================================================

        self.actor = Actor(actor_state_dim).to(self.device)

        self.critic = Critic(critic_state_dim).to(self.device)

        # =================================================
        # Optimizers
        # =================================================

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=lr_actor,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=lr_critic,
        )

        # =================================================
        # Reward Parameters
        # =================================================

        self.alpha = alpha
        self.beta = beta
        self.mu = mu

        self.gamma = gamma

        # =================================================
        # Gradient Accumulation Configuration
        # =================================================

        self.gradient_chunk_size = max(
            1,
            gradient_chunk_size,
        )

        # =================================================
        # Gradient Statistics History
        #
        # Store the accumulated Actor gradients
        # from previous episodes.
        # =================================================

        self.gradient_history = []

        self.gradient_variance_window = 100

    # =====================================================
    # TERMINAL REWARD
    # =====================================================

    def calculate_terminal_reward(self, env):
        """
        Calculate the final packing objective.

        Intermediate rewards are not included.
        """

        metrics = env.calculate_metrics()

        num_bins = metrics["bins_used"]

        utilization = metrics["utilization"]

        packing_density = metrics["packing_density"]

        stability = metrics["stability"]

        terminal_reward = (
            -num_bins
            + self.alpha * utilization
            + self.beta * packing_density
            + self.mu * stability
        )

        return float(terminal_reward), metrics

    # =====================================================
    # GENERATE EPISODE
    # =====================================================

    def generate_episode(self, env, episode=0):
        """
        Generate one complete episode.

        The Actor and Critic use env.get_state_vector().

        The dictionary returned by env.step() is ignored.

        Intermediate rewards are intentionally ignored.
        """

        # Reset environment.
        env.reset()

        states = []

        next_states = []

        log_probs = []

        dones = []

        done = False

        while not done:

            # ---------------------------------------------
            # Select action using the Actor.
            #
            # select_action returns:
            # action, log_prob, state_tensor
            # ---------------------------------------------

            action, log_prob, state_tensor = select_action(
                self.actor,
                env,
                self.device,
                episode,
            )

            # ---------------------------------------------
            # Execute selected action.
            #
            # The returned state is a dictionary.
            # The reward is ignored (NO EARLY FEEDBACK).
            # ---------------------------------------------

            _, _, done, info = env.step(action)

            # ---------------------------------------------
            # Validate action.
            # ---------------------------------------------

            if info.get("invalid_action", False):

                raise RuntimeError(
                    f"Invalid action encountered: {action}"
                )

            # ---------------------------------------------
            # Store current state.
            #
            # State before executing the action.
            # ---------------------------------------------

            states.append(
                state_tensor.detach()
            )

            # ---------------------------------------------
            # Get numerical next state.
            #
            # Do not convert env.step() observation.
            # Use get_state_vector() instead.
            # ---------------------------------------------

            next_state_vector = env.get_state_vector()

            next_state_tensor = torch.as_tensor(
                next_state_vector,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0).detach()

            next_states.append(
                next_state_tensor
            )

            # ---------------------------------------------
            # Store policy probability.
            # ---------------------------------------------

            log_probs.append(log_prob)

            # ---------------------------------------------
            # Store terminal indicator.
            # ---------------------------------------------

            dones.append(bool(done))

        # =================================================
        # Calculate Terminal Reward
        # =================================================

        terminal_reward, metrics = (
            self.calculate_terminal_reward(env)
        )

        return {
            "states": states,
            "next_states": next_states,
            "log_probs": log_probs,
            "dones": dones,
            "terminal_reward": terminal_reward,
            "metrics": metrics,
            "num_steps": len(states),
        }

    # =====================================================
    # UPDATE ACTOR AND CRITIC
    # =====================================================

    def update(self, episode_data):
        """
        Perform TD(0) Actor-Critic update.

        Intermediate rewards are always zero.

        TD target:

            target = 0 + gamma * V(next_state)

        Terminal target:

            target = terminal_reward

        Gradients are accumulated across chunks.
        Optimizer steps are performed once per episode.
        """

        states = episode_data["states"]

        next_states = episode_data["next_states"]

        log_probs = episode_data["log_probs"]

        dones = episode_data["dones"]

        terminal_reward = episode_data["terminal_reward"]

        num_steps = len(states)

        if num_steps == 0:

            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "gradient_variance": 0.0,
                "gradient_norm": 0.0,
            }

        # =================================================
        # Reset Gradients
        # =================================================

        self.actor_optimizer.zero_grad()

        self.critic_optimizer.zero_grad()

        actor_loss_value = 0.0

        critic_loss_value = 0.0

        # =================================================
        # Process Transitions in Chunks
        #
        # Gradients accumulate across all chunks.
        # =================================================

        for start in range(
            0,
            num_steps,
            self.gradient_chunk_size,
        ):

            end = min(
                start + self.gradient_chunk_size,
                num_steps,
            )

            chunk_actor_loss = 0.0

            chunk_critic_loss = 0.0

            for t in range(start, end):

                state = states[t]

                next_state = next_states[t]

                # -----------------------------------------
                # Current State Value
                # -----------------------------------------

                value = self.critic(state).squeeze()

                # -----------------------------------------
                # TD(0) Target
                # -----------------------------------------

                if dones[t]:

                    # Terminal transition:
                    # Use the final packing objective.

                    target = torch.tensor(
                        terminal_reward,
                        dtype=torch.float32,
                        device=self.device,
                    )

                else:

                    # Intermediate transition:
                    # r_t = 0 (NO EARLY FEEDBACK).
                    #
                    # Bootstrap using the next-state value.

                    with torch.no_grad():

                        next_value = (
                            self.critic(next_state).squeeze()
                        )

                    target = self.gamma * next_value

                # -----------------------------------------
                # TD Error
                # -----------------------------------------

                td_error = target - value

                # -----------------------------------------
                # Actor Loss
                # -----------------------------------------

                actor_loss = (
                    -log_probs[t] * td_error.detach()
                )

                # -----------------------------------------
                # Critic Loss
                # -----------------------------------------

                critic_loss = td_error.pow(2)

                chunk_actor_loss += actor_loss

                chunk_critic_loss += critic_loss

                actor_loss_value += (
                    actor_loss.detach().item()
                )

                critic_loss_value += (
                    critic_loss.detach().item()
                )

            # ---------------------------------------------
            # Normalize by total episode length.
            #
            # This creates average episode gradients.
            # ---------------------------------------------

            chunk_actor_loss = (
                chunk_actor_loss / num_steps
            )

            chunk_critic_loss = (
                chunk_critic_loss / num_steps
            )

            # Accumulate gradients.
            chunk_actor_loss.backward()

            chunk_critic_loss.backward()

        # =================================================
        # GRADIENT STATISTICS
        #
        # Calculate statistics using accumulated Actor
        # gradients over the FULL episode.
        # =================================================

        gradient_norm = 0.0

        gradient_variance = 0.0

        grads = []

        for parameter in self.actor.parameters():

            if parameter.grad is not None:

                grads.append(
                    parameter.grad.detach().flatten()
                )

        if len(grads) > 0:

            # ---------------------------------------------
            # Concatenate all Actor gradients.
            # ---------------------------------------------

            current_gradient = torch.cat(grads)

            # ---------------------------------------------
            # Gradient Norm
            #
            # L2 norm of the accumulated episode gradient.
            # ---------------------------------------------

            gradient_norm = (
                current_gradient.norm().item()
            )

            # ---------------------------------------------
            # Store gradient on CPU.
            # ---------------------------------------------

            current_gradient_cpu = (
                current_gradient.detach().cpu()
            )

            self.gradient_history.append(
                current_gradient_cpu
            )

            # ---------------------------------------------
            # Keep only the latest 100 episodes.
            # ---------------------------------------------

            if (
                len(self.gradient_history)
                > self.gradient_variance_window
            ):

                self.gradient_history.pop(0)

            # ---------------------------------------------
            # Gradient Variance
            #
            # Calculate variance across episodes
            # for every Actor parameter.
            #
            # Then calculate mean variance.
            # ---------------------------------------------

            if len(self.gradient_history) > 1:

                gradient_matrix = torch.stack(
                    self.gradient_history
                )

                gradient_variance = (
                    gradient_matrix
                    .var(
                        dim=0,
                        unbiased=False,
                    )
                    .mean()
                    .item()
                )

        # =================================================
        # Optimizer Step
        #
        # Update Actor and Critic once per episode.
        # =================================================

        self.actor_optimizer.step()

        self.critic_optimizer.step()

        # =================================================
        # Return Update Statistics
        # =================================================

        return {
            "actor_loss": (
                actor_loss_value / num_steps
            ),

            "critic_loss": (
                critic_loss_value / num_steps
            ),

            "gradient_variance": gradient_variance,

            "gradient_norm": gradient_norm,
        }

    # =====================================================
    # TRAIN ONE EPISODE
    # =====================================================

    def train_episode(self, env, episode=0):
        """
        Generate an episode, update the networks,
        and return training metrics.
        """

        episode_data = self.generate_episode(
            env,
            episode,
        )

        update_metrics = self.update(
            episode_data
        )

        metrics = episode_data["metrics"]

        return {

            "reward": episode_data["terminal_reward"],

            "bins": metrics["bins_used"],

            "utilization": metrics["utilization"],

            "packing_density": metrics["packing_density"],

            "stability": metrics["stability"],

            "actor_loss": update_metrics["actor_loss"],

            "critic_loss": update_metrics["critic_loss"],

            "gradient_variance": (
                update_metrics["gradient_variance"]
            ),

            "gradient_norm": (
                update_metrics["gradient_norm"]
            ),
        }
