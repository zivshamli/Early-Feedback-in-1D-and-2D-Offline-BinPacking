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
            device=None
    ):


        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"


        self.device = torch.device(device)


        print("Using device:", self.device)


        self.alpha = alpha
        self.gamma = gamma
        self.gradient_history = []
        self.gradient_variance_window = 100


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



    def calculate_terminal_reward(self, env):


        num_bins = len(env.bins)

        utilization = env._calculate_utilization()


        reward = (
            -num_bins
            +
            self.alpha * utilization
        )


        return reward



    def generate_episode(self, env, episode):


        env.reset()

        trajectory = []

        done = False

        count=0

        while not done:

            action, log_prob, state_tensor = select_action(
                self.actor,
                env,
                self.device
                ,episode
            )

            value = self.critic(
                state_tensor
            )



            next_state, reward, done, info = env.step(
                action
            )
        



            trajectory.append(
                {
                    
                    "log_prob": log_prob,
                    "state": state_tensor
                }
            )
            count+=1



        terminal_reward = self.calculate_terminal_reward(env)


        return trajectory, terminal_reward




    def update(self, trajectory, reward):

        actor_losses = []
        critic_losses = []

        G = torch.tensor(
            reward,
            dtype=torch.float32,
            device=self.device
        )

        for step in trajectory:

            state = step["state"]
            value = self.critic(state)
            log_prob = step["log_prob"]

            advantage = G - value.detach()

            actor_losses.append(
                -log_prob * advantage
            )

            critic_losses.append(
                nn.functional.mse_loss(
                    value.squeeze(),
                    G
                )
            )

        actor_loss = torch.stack(actor_losses).mean()
        critic_loss = torch.stack(critic_losses).mean()

        ##################################################
        # Actor Update
        ##################################################

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        grads = []

        for p in self.actor.parameters():

            if p.grad is not None:
                grads.append(
                    p.grad.detach().flatten()
                )

        if len(grads) > 0:

            current_gradient = torch.cat(grads)

            # Gradient norm for current update
            gradient_norm = current_gradient.norm().item()

            # Store gradient on CPU to avoid GPU memory growth
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
                    .var(dim=0, unbiased=False)
                    .mean()
                    .item()
                )

            else:

                gradient_variance = 0.0

        else:

            gradient_variance = 0.0
            gradient_norm = 0.0

        self.actor_optimizer.step()

        ##################################################
        # Critic Update
        ##################################################

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        self.critic_optimizer.step()

        actor_loss_value = actor_loss.item()
        critic_loss_value = critic_loss.item()

        for step in trajectory:
            step["log_prob"] = None
            step["value"] = None

        del trajectory
        del actor_loss
        del critic_loss
        del G

        torch.cuda.empty_cache()

        return (
            actor_loss_value,
            critic_loss_value,
            gradient_variance,
            gradient_norm
        )