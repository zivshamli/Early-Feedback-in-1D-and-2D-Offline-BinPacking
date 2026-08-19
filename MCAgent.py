import torch
import torch.nn as nn

from Actor import Actor, select_action
from Critic import Critic


class MonteCarloActorCritic:

    def __init__(
        self,
        actor_state_dim,
        critic_state_dim,
        alpha=100,
        gamma=1.0,
        lr_actor=1e-4,
        lr_critic=1e-3,
        device=None,
        gradient_chunk_size=20
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
        # Chunk size
        #
        # Number of transitions processed by each backward.
        #
        # IMPORTANT:
        # This does NOT mean optimizer update.
        #
        # Optimizer.step() is still done only once per episode.
        # =================================================

        self.gradient_chunk_size = gradient_chunk_size

        print(
            "MC gradient chunk size:",
            self.gradient_chunk_size
        )

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
    # We keep the ORIGINAL select_action().
    #
    # We also keep the ORIGINAL log_prob.
    #
    # save_on_cpu() moves tensors saved for backward to CPU.
    #
    # This reduces GPU memory without changing the policy,
    # sampling, reward, or loss definition.
    # =========================================================

    def generate_episode(
        self,
        env,
        episode
    ):

        env.reset()

        trajectory = []

        done = False

        # =====================================================
        # Save autograd tensors on CPU
        #
        # This is the key memory optimization.
        # =====================================================

        if self.device.type == "cuda":

            graph_context = torch.autograd.graph.save_on_cpu(
                pin_memory=False
            )

        else:

            graph_context = torch.enable_grad()

        with graph_context:

            while not done:

                # =============================================
                # ORIGINAL select_action
                # =============================================

                action, log_prob, state_tensor = select_action(
                    self.actor,
                    env,
                    self.device,
                    episode
                )

                # =============================================
                # IMPORTANT:
                #
                # The original code calculates value here,
                # but does not use it.
                #
                # Therefore we do NOT calculate it.
                #
                # This does not change the MC algorithm.
                # =============================================

                # =============================================
                # Environment step
                # =============================================

                _, _, done, info = env.step(
                    action
                )

                # =============================================
                # Store the ORIGINAL log_prob and state.
                #
                # We intentionally DO NOT detach them.
                # =============================================

                trajectory.append(
                    {
                        "log_prob": log_prob,
                        "state": state_tensor
                    }
                )

        # =====================================================
        # Terminal reward
        # =====================================================

        terminal_reward = (
            self.calculate_terminal_reward(env)
        )

        return (
            trajectory,
            terminal_reward
        )

    # =========================================================
    # Monte Carlo Update
    #
    # Original algorithm:
    #
    # G = terminal reward
    #
    # advantage = G - V(s).detach()
    #
    # actor_loss =
    #     mean(-log_prob * advantage)
    #
    # critic_loss =
    #     mean(MSE(V(s), G))
    #
    # We calculate the same losses in chunks.
    #
    # The gradients accumulate across chunks.
    #
    # optimizer.step() happens ONLY ONCE.
    # =========================================================

    def update(
        self,
        trajectory,
        reward
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
        # Monte Carlo Return
        #
        # Same as original.
        # =====================================================

        G = torch.tensor(
            reward,
            dtype=torch.float32,
            device=self.device
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

        # =====================================================
        # Loss values for logging
        # =====================================================

        actor_loss_total = 0.0
        critic_loss_total = 0.0

        # =====================================================
        # Process episode in chunks
        # =====================================================

        for chunk_start in range(
            0,
            num_steps,
            self.gradient_chunk_size
        ):

            chunk_end = min(
                chunk_start
                + self.gradient_chunk_size,
                num_steps
            )

            chunk = trajectory[
                chunk_start:chunk_end
            ]

            # Number of transitions in this chunk

            chunk_size = len(chunk)

            # =================================================
            # IMPORTANT:
            #
            # Original loss:
            #
            # mean(all transition losses)
            #
            # Therefore each chunk must use:
            #
            # sum(chunk losses) / num_steps
            #
            # NOT:
            #
            # mean(chunk losses)
            #
            # This preserves the original scaling.
            # =================================================

            actor_chunk_losses = []
            critic_chunk_losses = []

            # =================================================
            # Build losses for current chunk
            # =================================================

            for step in chunk:

                # =============================================
                # Original state
                # =============================================

                state = step["state"]

                # =============================================
                # Original log probability
                #
                # This is the ORIGINAL log_prob produced by
                # select_action().
                # =============================================

                log_prob = step["log_prob"]

                # =============================================
                # Critic forward
                # =============================================

                value = self.critic(
                    state
                )

                # =============================================
                # Monte Carlo Advantage
                #
                # EXACTLY original.
                # =============================================

                advantage = (
                    G
                    - value.detach()
                )

                # =============================================
                # Actor loss
                # =============================================

                actor_loss_step = (
                    -log_prob
                    * advantage
                )

                # =============================================
                # Critic loss
                # =============================================

                critic_loss_step = (
                    nn.functional.mse_loss(
                        value.squeeze(),
                        G
                    )
                )

                actor_chunk_losses.append(
                    actor_loss_step
                )

                critic_chunk_losses.append(
                    critic_loss_step
                )

                # =============================================
                # Numerical logging
                # =============================================

                actor_loss_total += (
                    actor_loss_step.detach().item()
                )

                critic_loss_total += (
                    critic_loss_step.detach().item()
                )

                # =============================================
                # IMPORTANT:
                #
                # We cannot delete log_prob/state here
                # manually because they are needed by the
                # backward graph of actor_loss_step.
                #
                # They are released after backward().
                # =============================================

            # =================================================
            # Chunk losses
            #
            # SUM / TOTAL NUMBER OF STEPS
            #
            # This gives the same scaling as:
            #
            # torch.stack(actor_losses).mean()
            #
            # from the original implementation.
            # =================================================

            actor_chunk_loss = (
                torch.stack(
                    actor_chunk_losses
                ).sum()
                / num_steps
            )

            critic_chunk_loss = (
                torch.stack(
                    critic_chunk_losses
                ).sum()
                / num_steps
            )

            # =================================================
            # Backward for this chunk
            #
            # Gradients accumulate.
            #
            # NO optimizer.step() here.
            # =================================================

            actor_chunk_loss.backward()

            critic_chunk_loss.backward()

            # =================================================
            # Free chunk references
            # =================================================

            del actor_chunk_losses
            del critic_chunk_losses
            del actor_chunk_loss
            del critic_chunk_loss
            del chunk

        # =====================================================
        # At this point:
        #
        # actor.grad contains accumulated gradient over the
        # ENTIRE episode.
        #
        # critic.grad contains accumulated gradient over the
        # ENTIRE episode.
        # =====================================================

        # =====================================================
        # Gradient statistics
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

            self.gradient_history.append(
                current_gradient.detach().cpu()
            )

            # -------------------------------------------------
            # Keep latest 100 updates
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

                del gradient_matrix

            del current_gradient

        # =====================================================
        # ONE Actor optimizer update
        # =====================================================

        self.actor_optimizer.step()

        # =====================================================
        # ONE Critic optimizer update
        # =====================================================

        self.critic_optimizer.step()

        # =====================================================
        # Mean losses
        #
        # Same values as original:
        #
        # torch.stack(losses).mean()
        # =====================================================

        actor_loss_value = (
            actor_loss_total
            / num_steps
        )

        critic_loss_value = (
            critic_loss_total
            / num_steps
        )

        # =====================================================
        # Free trajectory
        # =====================================================

        del trajectory
        del G

        # =====================================================
        # CUDA cache
        #
        # Only once per episode.
        # =====================================================

        if self.device.type == "cuda":

            torch.cuda.empty_cache()

        # =====================================================
        # Same return interface
        # =====================================================

        return (
            actor_loss_value,
            critic_loss_value,
            gradient_variance,
            gradient_norm
        )