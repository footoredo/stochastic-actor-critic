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
# torch.autograd.set_detect_anomaly(True)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

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


# class BatchBayes(nn.Module):
#     def forward(self, p, s, a):
#         pp = p * torch.bmm(s, a)
#         # pp = torch.squeeze(pp, 2)
#         # print(pp.size())
#         # pp = p
#         # print(pp.size())
#         pp = torch.squeeze(pp, -1)
#         return pp / torch.sum(p, -1)

def batch_bayes(p, s, a):
    pp = p * torch.bmm(s, a)
    # pp = torch.squeeze(pp, 2)
    # print(pp.size())
    # pp = p
    # print(pp.size())
    pp = torch.squeeze(pp, -1)
    # print(p, s, a, pp, torch)
    return pp / torch.sum(pp, -1)

# batch_bayes = BatchBayes()


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
        y = torch.matmul(atk_type, ts(p)).reshape([-1, n_slots, n_slots])
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


class Pack(nn.Module):
    def __init__(self, samples, p=2):
        super(Pack, self).__init__()
        self.samples = []
        self.v_dim = samples[0][1].shape[0]
        for k, v in samples:
            self.samples.append((ts(k), ts(v)))
        self.p = p

    def forward(self, b):
        sum_w = torch.zeros(b.size()[0])
        sum_v = torch.zeros([b.size()[0], self.v_dim])
        for k, v in self.samples:
            dis = torch.norm(k - b, dim=1)
            # print(b.size())
            w = torch.div(torch.ones_like(dis), torch.clamp(torch.pow(dis, self.p), min=1e-6))
            sum_w += w
            sum_v += torch.ger(w, v)
        return sum_v / sum_w


sss = """-0.1931300893	0.961026482	0.7378740233
-0.1030614502	0.9524352491	0.7353386552
-0.2182349191	0.9660896315	0.7358214949
-0.1975423002	0.96085326	0.7358739475
-0.1973181798	0.9607963711	0.7358709263
-0.1978641531	0.9614468715	0.7358740167
-0.8089985514	1.73586039	0.7358769881
-0.8093295141	1.736104729	0.7358607003
-0.8094510535	1.736082369	0.7358952676
-0.8095254252	1.735848667	0.7358598875
-0.8102910921	1.716189773	0.7317437907"""

sss = np.array(list(map(float, sss.split()))).reshape((11, n_types + 1))
atk_vn = []
for t in range(n_types):
    samples = []
    for p in range(11):
        samples.append((np.array([p / 10, (10 - p) / 10]), sss[p, t].reshape(1)))
    atk_vn.append(Pack(samples))

samples = []
for p in range(11):
    samples.append((np.array([p / 10, (10 - p) / 10]), sss[p, n_types].reshape(1)))
dfd_vn = Pack(samples)


# print(atk_vn[0](ts([[0.6, 0.4]])))


class Model(nn.Module):
    def __init__(self, atk_actor, dfd_actor, critic, is_atk, vn=None):
        super(Model, self).__init__()
        self.atk_actor = atk_actor
        self.dfd_actor = dfd_actor
        self.critic = critic
        self.is_atk = is_atk
        self.vn = vn

    def forward(self, prior, r):
        batch_size = prior.size()[0]
        dfd_ob = torch.cat((prior, r), 1)
        dfd_prob = self.dfd_actor(dfd_ob)

        atk_ac = torch.zeros((batch_size, n_slots))
        dfd_ac = torch.zeros((batch_size, n_slots))
        atk_tp = torch.zeros((batch_size, n_types))

        y = torch.zeros((batch_size, 1))

        atk_s = torch.zeros((n_types, batch_size, n_slots))

        for t in range(n_types):
            atk_tp[:, t] = 1.
            atk_ob = torch.cat((prior, r, atk_tp), 1)
            atk_prob = self.atk_actor(atk_ob)
            atk_s[t] = atk_prob
            atk_tp[:, t] = 0.

        atk_s = torch.transpose(atk_s, 0, 1)

        for t in range(n_types):
            atk_tp[:, t] = 1.
            atk_prob = atk_s[:, t, :]
            for a in range(n_slots):
                atk_ac = torch.zeros((batch_size, n_slots))
                atk_ac[:, a] = 1.
                new_prior = batch_bayes(prior.unsqueeze(-1), atk_s, atk_ac.unsqueeze(-1))
                # print(prior, atk_s, atk_ac, new_prior)
                for b in range(n_slots):
                    dfd_ac[:, b] = 1.

                    c = self.critic(prior, r, atk_tp, atk_prob, atk_ac, dfd_prob, dfd_ac)

                    cc = ts(0.)
                    if self.vn is not None:
                        if type(self.vn) == list:
                            cc = self.vn[t](new_prior)
                        else:
                            cc = self.vn(new_prior)

                    # print(c, cc)
                    c += cc.detach()

                    # print(c)

                    if self.is_atk:
                        y += c * atk_prob[:, a].unsqueeze(-1) * dfd_prob[:, b].unsqueeze(-1) / n_types
                    else:
                        y += c * atk_prob[:, a].unsqueeze(-1) * dfd_prob[:, b].unsqueeze(-1) * prior[:, t].unsqueeze(-1)

                    dfd_ac[:, b] = 0.
                # atk_ac[:, a] = 0.
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
    idx = ts(idx)
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
        return self.actor(ts([ob])).data[0].cpu().numpy()

    def act(self, ob):
        s = self.strategy(ob)
        return sample_action(s)

    def prob(self, ob, a):
        s = self.strategy(ob)
        return s[a]


for i in range(5000):

    atk_obs, dfd_obs = collect(1)
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

        if i % 100 == 99:
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

            env.assess_strategies((Strategy(avg_atk_actor), Strategy(avg_dfd_actor)))

