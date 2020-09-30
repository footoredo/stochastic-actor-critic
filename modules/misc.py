import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import ts


class Strategy(nn.Module):
    def __init__(self, n_actions, n_types=1, init=None):
        super(Strategy, self).__init__()
        self.n_actions = n_actions
        self.n_types = n_types
        shape = (n_types, n_actions) if n_types > 1 else n_actions
        # print("init:", init)
        data = init if init is not None else np.zeros(shape)
        self.logits = nn.Parameter(torch.tensor(data, dtype=torch.float), requires_grad=True)

    def forward(self):
        # strategy = self.logits
        strategy = F.softmax(self.logits, dim=-1)
        # strategy = torch.clamp(self.logits, 0.0, 1.0)
        # strategy = torch.clamp(self.logits, min=0.)
        # if strategy.sum() > 0.:
        #     return strategy / strategy.sum()
        # else:
        #     return F.softmax(self.logits, dim=-1)

        return strategy


class Bayes(nn.Module):
    def __init__(self, n_types, n_actions):
        super(Bayes, self).__init__()
        self.n_types = n_types
        self.n_actions = n_actions

    def forward(self, belief, strategy, action):
        prob = torch.gather(strategy, 1, action.repeat(self.n_types, 1)).transpose(0, 1)  # [batch, t]
        new_belief = prob * belief.unsqueeze(0)
        return new_belief / new_belief.sum(-1, keepdim=True)


class BayesFast(nn.Module):
    def __init__(self, n_types, n_actions):
        super(BayesFast, self).__init__()
        self.n_types = n_types
        self.n_actions = n_actions

    def forward(self, belief, strategy, action):
        prob = strategy[:, action]
        new_belief = prob * belief
        new_belief = new_belief / new_belief.sum()
        return new_belief


if __name__ == "__main__":
    bayes = Bayes(2, 2)
    print(bayes(ts([0.5, 0.5]), ts([[0.1, 0.9], [0.9, 0.1]]), torch.tensor([1, 0, 1])))

    bayes_fast = BayesFast(2, 2)
    print(bayes_fast(ts([0.5, 0.5]), ts([[0.1, 0.9], [0.9, 0.1]]), 1))
    # tensor([[0.9000, 0.1000],
    #         [0.1000, 0.9000],
    #         [0.9000, 0.1000]])
