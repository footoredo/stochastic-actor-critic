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
args = parser.parse_args()


env = SecurityGame(n_slots=2, n_types=2, prior=np.array([0.6, 0.4]), n_rounds=1, seed=args.seed)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


n_rounds = 1
n_types = 2
n_slots = 2

hid_size = 64


def ts(v):
    return torch.tensor(v, dtype=torch.float)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.affine1 = nn.Linear(n_types + n_rounds + n_types + n_slots * 4, hid_size)
        self.affine2 = nn.Linear(hid_size, 1)

    def forward(self, prior, r, atk_type, atk_prob, atk_ac, dfd_prob, dfd_ac):
        x = torch.cat((prior, r, atk_type, atk_prob, atk_ac, dfd_prob, dfd_ac), 1)
        # print(x.size(), x.data[0])
        y = F.relu(self.affine1(x))
        y = self.affine2(y)
        return y

    def average(self, other, k):
        self.affine1.weight.data = (1 - k) * self.affine1.weight.data + k * other.affine1.weight.data
        self.affine1.bias.data = (1 - k) * self.affine1.bias.data + k * other.affine1.bias.data
        self.affine2.weight.data = (1 - k) * self.affine2.weight.data + k * other.affine2.weight.data
        self.affine2.bias.data = (1 - k) * self.affine2.bias.data + k * other.affine2.bias.data

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
    def __init__(self, atk_actor, dfd_actor, critic):
        super(Model, self).__init__()
        self.atk_actor = atk_actor
        self.dfd_actor = dfd_actor
        self.critic = critic

    def forward(self, prior, r, atk_type):
        atk_ob = torch.cat((prior, r, atk_type), 1)
        dfd_ob = torch.cat((prior, r), 1)
        atk_prob = self.atk_actor(atk_ob)
        dfd_prob = self.dfd_actor(dfd_ob)

        atk_ac = torch.zeros_like(atk_prob)
        dfd_ac = torch.zeros_like(dfd_prob)

        batch_size = atk_ac.size()[0]

        y = torch.zeros((batch_size, 1))

        for a in range(n_slots):
            atk_ac[:, a] = 1.
            for b in range(n_slots):
                dfd_ac[:, b] = 1.

                c = self.critic(prior, r, atk_type, atk_prob, atk_ac, dfd_prob, dfd_ac)
                y += c * atk_prob[:, a].unsqueeze(-1) * dfd_prob[:, b].unsqueeze(-1)

                dfd_ac[:, b] = 0.
            atk_ac[:, a] = 0.

        return y


critic_lr = 3e-2
actor_lr = 1e-3

atk_actor = Actor(n_types + n_rounds + n_types)
avg_atk_actor = Actor(n_types + n_rounds + n_types)
dfd_actor = Actor(n_types + n_rounds)
avg_dfd_actor = Actor(n_types + n_rounds)
atk_critic = Critic()
dfd_critic = Critic()
target_atk_critic = Critic()
target_dfd_critic = Critic()
cnt_c = 1
target_atk_critic.average(atk_critic, 1. / cnt_c)
target_dfd_critic.average(dfd_critic, 1. / cnt_c)
atk_model = Model(atk_actor, dfd_actor, target_atk_critic)
dfd_model = Model(atk_actor, dfd_actor, target_dfd_critic)

cnt = 1
avg_atk_actor.average(atk_actor, 1. / cnt)
avg_dfd_actor.average(dfd_actor, 1. / cnt)

atk_critic_optimizer = optim.Adam(atk_critic.parameters(), lr=critic_lr)
dfd_critic_optimizer = optim.Adam(dfd_critic.parameters(), lr=critic_lr)
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
    atk_probs = []
    dfd_probs = []
    atk_acs = []
    dfd_acs = []
    atk_rews = []
    dfd_rews = []
    for i in range(n):
        atk_obs.append(atk_ob)
        dfd_obs.append(dfd_ob)
        atk_prob = atk_actor(torch.tensor([atk_ob], dtype=torch.float))[0]
        dfd_prob = dfd_actor(torch.tensor([dfd_ob], dtype=torch.float))[0]
        atk_ac = sample_action(atk_prob)
        dfd_ac = sample_action(dfd_prob)
        obs, rews, done, _ = env.step([atk_ac, dfd_ac])

        atk_ob, dfd_ob = obs
        atk_rew, dfd_rew = rews

        atk_probs.append(atk_prob)
        dfd_probs.append(dfd_prob)
        atk_acs.append(atk_ac)
        dfd_acs.append(dfd_ac)
        atk_rews.append(atk_rew)
        dfd_rews.append(dfd_rew)

        if done:
            atk_ob, dfd_ob = env.reset()

    return atk_obs, dfd_obs, atk_probs, dfd_probs, atk_acs, dfd_acs, atk_rews, dfd_rews


critic_criterion = nn.MSELoss()

for i in range(5000):
    print("--- iter {} ---".format(i))

    atk_obs, dfd_obs, atk_probs, dfd_probs, atk_acs, dfd_acs, atk_rews, dfd_rews = collect(500)
    priors, rs, atk_types = torch.split(ts(atk_obs), [n_types, n_rounds, n_types], dim=1)
    atk_probs = torch.stack(atk_probs)
    dfd_probs = torch.stack(dfd_probs)
    atk_acs = one_hot(n_slots, atk_acs)
    dfd_acs = one_hot(n_slots, dfd_acs)

    # cnt_c += 1

    atk_critic_optimizer.zero_grad()
    atk_critic_result = atk_critic(priors, rs, atk_types, atk_probs, atk_acs, dfd_probs, dfd_acs)
    # print(atk_critic_result)
    print(atk_rews[0], atk_critic_result.data[0])
    atk_critic_loss = critic_criterion(ts(atk_rews).unsqueeze(-1), atk_critic_result)
    # if i < 2000:
    atk_critic_loss.backward(retain_graph=True)
    atk_critic_optimizer.step()

    target_atk_critic.average(atk_critic, 1. / cnt_c)

    dfd_critic_optimizer.zero_grad()
    dfd_critic_result = dfd_critic(priors, rs, atk_types, atk_probs, atk_acs, dfd_probs, dfd_acs)
    print(dfd_rews[0], dfd_critic_result.data[0])
    dfd_critic_loss = critic_criterion(ts(dfd_rews).unsqueeze(-1), dfd_critic_result)
    # if i < 2000:
    dfd_critic_loss.backward(retain_graph=True)
    dfd_critic_optimizer.step()

    target_dfd_critic.average(dfd_critic, 1. / cnt_c)

    print(atk_critic_loss.data)
    print(dfd_critic_loss.data)

    if i >= 0:
        atk_actor_optimizer.zero_grad()
        atk_actor_loss = -atk_model(priors, rs, atk_types).mean()
        atk_actor_loss.backward(retain_graph=True)
        atk_actor_optimizer.step()

        print(atk_actor_loss.data)
        print(atk_actor(ts([[0.5, 0.5, 1., 1., 0.]])).data[0],
              atk_actor(ts([[0.5, 0.5, 1., 0., 1.]])).data[0])

        dfd_actor_optimizer.zero_grad()
        dfd_actor_loss = -dfd_model(priors, rs, atk_types).mean()
        dfd_actor_loss.backward(retain_graph=True)
        dfd_actor_optimizer.step()

        print(dfd_actor_loss.data)
        print(dfd_actor(ts([[0.5, 0.5, 1.]])).data[0])

        # print(dfd_actor.affine1.weight.grad)

        print("avg")
        cnt += 1
        avg_atk_actor.average(atk_actor, 1. / cnt)
        avg_dfd_actor.average(dfd_actor, 1. / cnt)

        for i in range(11):
            p0 = i / 10.
            p1 = 1 - i / 10.
            print(avg_atk_actor(ts([[p0, p1, 1., 1., 0.]])).data[0],
                  avg_atk_actor(ts([[p0, p1, 1., 0., 1.]])).data[0])
            print(avg_dfd_actor(ts([[p0, p1, 1.]])).data[0])