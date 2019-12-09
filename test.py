import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.affine1 = nn.Linear(1, 2, bias=False)
        self.output = None

    def forward(self, x):
        self.output = F.softmax(self.affine1(x), dim=-1)
        return self.output


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.affine1 = nn.Linear(4, 1, bias=False)

    def forward(self, x):
        return self.affine1(x)


class ActorLoss(nn.Module):
    def __init__(self, actor, critic):
        super(ActorLoss, self).__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, x):
        y = self.actor(x)
        y0 = self.critic(torch.cat((y, torch.tensor([[1., 0.]])), 1))
        y1 = self.critic(torch.cat((y, torch.tensor([[0., 1.]])), 1))
        return y0 * y[:, 0] + y1 * y[:, 1]


actor = Actor()
critic = Critic()
actor_loss = ActorLoss(actor, critic)

actual_actor_loss = actor_loss(torch.tensor([[5.]]))
actual_actor_loss.backward()
print(actor.affine1.weight.grad)
print(actor.affine1.weight.data)
print(actor.output.data)
print(critic.affine1.weight.grad)
print(critic.affine1.weight.data)
print(actual_actor_loss.data)

