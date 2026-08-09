import torch.nn as nn


class Critic(nn.Module):

    def __init__(self,state_dim):

        super().__init__()


        self.network=nn.Sequential(

            nn.Linear(state_dim,128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,1)
        )


    def forward(self,state):

        return self.network(state)