import argparse
import gym
from env.security_game import SecurityGame
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Cart Pole

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--prior', type=float, nargs='+')
args = parser.parse_args()


env = SecurityGame(n_slots=2, n_types=2, prior=np.array(args.prior), n_rounds=1, seed=args.seed, random_prior=False)
payoff = env.payoff
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


n_rounds = 1
n_types = 2
n_slots = 2

hid_size = 32


def ts(v):
    return torch.tensor(v, dtype=torch.float)


class Critic(nn.Module):
    def __init__(self, payoff):
        super(Critic, self).__init__()
        # self.affine1 = nn.Linear(n_types + n_rounds + n_types + n_slots * 4, hid_size)
        # self.affine2 = nn.Linear(hid_size, 1)
        self.payoff = payoff  # t * n * n

    def forward(self, prior, r, atk_type, atk_prob, atk_ac, dfd_prob, dfd_ac):
        # x = torch.cat((prior, r, atk_type, atk_prob, atk_ac, dfd_prob, dfd_ac), 1)
        # # print(x.size(), x.data[0])
        # y = F.relu(self.affine1(x))
        # y = self.affine2(y)

        p = np.reshape(self.payoff, (n_types, n_slots * n_slots))
        y = torch.matmul(atk_type, torch.tensor(p)).reshape([-1, n_slots, n_slots])
        y = torch.bmm(atk_ac.reshape([-1, 1, n_slots]), y).reshape([-1, n_slots, 1])
        y = torch.bmm(dfd_ac.reshape([-1, 1, n_slots]), y).reshape([-1, 1])

        return y


class Actor(nn.Module):
    def __init__(self, ob_len):
        super(Actor, self).__init__()
        self.affine1 = nn.Linear(ob_len, hid_size)
        self.affine2 = nn.Linear(hid_size, n_slots)

    def forward(self, x):
        y = F.relu(self.affine1(x))
        y = self.affine2(y)
        y = F.softmax(y, dim=-1)
        return y

    def average(self, other, k):
        self.affine1.weight.data = (1 - k) * self.affine1.weight.data + k * other.affine1.weight.data
        self.affine1.bias.data = (1 - k) * self.affine1.bias.data + k * other.affine1.bias.data
        self.affine2.weight.data = (1 - k) * self.affine2.weight.data + k * other.affine2.weight.data
        self.affine2.bias.data = (1 - k) * self.affine2.bias.data + k * other.affine2.bias.data


class Model(nn.Module):
    def __init__(self, atk_actor, dfd_actor, critic, is_atk):
        super(Model, self).__init__()
        self.atk_actor = atk_actor
        self.dfd_actor = dfd_actor
        self.critic = critic
        self.is_atk = is_atk

    def forward(self, prior, r):
        dfd_ob = torch.cat((prior, r), 1)
        dfd_prob = self.dfd_actor(dfd_ob)

        atk_ac = torch.zeros_like(dfd_prob)
        dfd_ac = torch.zeros_like(dfd_prob)
        atk_tp = torch.zeros_like(prior)

        batch_size = atk_ac.size()[0]

        y = torch.zeros((batch_size, 1))

        for t in range(n_types):
            atk_tp[:, t] = 1.
            atk_ob = torch.cat((prior, r, atk_tp), 1)
            atk_prob = self.atk_actor(atk_ob)
            for a in range(n_slots):
                atk_ac[:, a] = 1.
                for b in range(n_slots):
                    dfd_ac[:, b] = 1.

                    c = self.critic(prior, r, atk_tp, atk_prob, atk_ac, dfd_prob, dfd_ac)
                    if self.is_atk:
                        y += c * atk_prob[:, a].unsqueeze(-1) * dfd_prob[:, b].unsqueeze(-1) / n_types
                    else:
                        y += c * atk_prob[:, a].unsqueeze(-1) * dfd_prob[:, b].unsqueeze(-1) * prior[:, t].unsqueeze(-1)

                    dfd_ac[:, b] = 0.
                atk_ac[:, a] = 0.
            atk_tp[:, t] = 0.
        return y


critic_lr = 3e-2
actor_lr = 1e-3

atk_actor = Actor(n_types + n_rounds + n_types)
avg_atk_actor = Actor(n_types + n_rounds + n_types)
dfd_actor = Actor(n_types + n_rounds)
avg_dfd_actor = Actor(n_types + n_rounds)
atk_critic = Critic(payoff[:, :, :, 0])
dfd_critic = Critic(payoff[:, :, :, 1])
atk_model = Model(atk_actor, dfd_actor, atk_critic, True)
dfd_model = Model(atk_actor, dfd_actor, dfd_critic, False)

cnt = 1
avg_atk_actor.average(atk_actor, 1. / cnt)
avg_dfd_actor.average(dfd_actor, 1. / cnt)

atk_actor_optimizer = optim.Adam(atk_actor.parameters(), lr=actor_lr, betas=(0.0, 0.999))
dfd_actor_optimizer = optim.Adam(dfd_actor.parameters(), lr=actor_lr, betas=(0.0, 0.999))


def sample_action(prob):
    return np.random.choice(range(n_slots), p=prob.detach().numpy())


def one_hot(n, idx):
    idx = torch.tensor(idx)
    x = torch.zeros(list(idx.size()) + [n])
    x.scatter_(-1, idx.unsqueeze(-1), 1.)
    return x


def collect(n):
    atk_ob, dfd_ob = env.reset()
    atk_obs = []
    dfd_obs = []
    for i in range(n):
        atk_obs.append(atk_ob)
        dfd_obs.append(dfd_ob)
        atk_ob, dfd_ob = env.reset()

    return atk_obs, dfd_obs


class Strategy(object):
    def __init__(self, actor):
        self.actor = actor

    def strategy(self, ob):
        return self.actor(torch.Tensor([ob])).data[0].numpy()

    def act(self, ob):
        s = self.strategy(ob)
        return sample_action(s)

    def prob(self, ob, a):
        s = self.strategy(ob)
        return s[a]


for i in range(50000):

    atk_obs, dfd_obs = collect(100)
    priors, rs, atk_types = torch.split(ts(atk_obs), [n_types, n_rounds, n_types], dim=1)

    # cnt_c += 1

    if i >= 0:
        atk_actor_optimizer.zero_grad()
        atk_actor_loss = -atk_model(priors, rs).mean()
        atk_actor_loss.backward(retain_graph=True)
        atk_actor_optimizer.step()

        # print(atk_actor_loss.data)
        # print(atk_actor(ts([[0.5, 0.5, 1., 1., 0.]])).data[0],
        #       atk_actor(ts([[0.5, 0.5, 1., 0., 1.]])).data[0])

        dfd_actor_optimizer.zero_grad()
        dfd_actor_loss = -dfd_model(priors, rs).mean()
        dfd_actor_loss.backward(retain_graph=True)
        dfd_actor_optimizer.step()

        # for param_group in atk_actor_optimizer.param_groups:
        #     param_group['lr'] = actor_lr / np.sqrt((i + 1) / 10 + 1)
        #
        # for param_group in dfd_actor_optimizer.param_groups:
        #     param_group['lr'] = actor_lr / np.sqrt((i + 1) / 10 + 1)

        # print(dfd_actor_loss.data)
        # print(dfd_actor(ts([[0.5, 0.5, 1.]])).data[0])

        # print(dfd_actor.affine1.weight.grad)

        cnt += 1
        avg_atk_actor.average(atk_actor, 1. / cnt)
        avg_dfd_actor.average(dfd_actor, 1. / cnt)

        if i % 10 == 9:
            print("--- iter {} ---".format(i))
            print("avg")
            # for p in range(11):
                # p = 9
            # p0 = p / 10.
            # p1 = (10 - p) / 10.
            p0 = args.prior[0]
            p1 = args.prior[1]
            print(p0, p1)
            print(avg_atk_actor(ts([[p0, p1, 1., 1., 0.]])).data[0],
                  avg_atk_actor(ts([[p0, p1, 1., 0., 1.]])).data[0])
            print(avg_dfd_actor(ts([[p0, p1, 1.]])).data[0])

            env.assess_strategies((Strategy(atk_actor), Strategy(dfd_actor)))