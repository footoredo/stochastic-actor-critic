import argparse
from env.security_game import SecurityGame
from env.demo_game import DemoGame
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.interpolate import CubicSpline, interp1d, Akima1DInterpolator

from multiprocessing import Pool
from functools import partial

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from utils.utils import *
from modules.pack import Interp1dPack
from modules.misc import *


def parse_args():
    parser = get_parser("mmmm")
    return parser.parse_args()


class Value(nn.Module):
    def __init__(self, n_actions, n_types, payoff, vn, is_atk, ratio):
        super(Value, self).__init__()
        self.n_actions = n_actions
        self.n_types = n_types
        self.bayes = Bayes(n_types, n_actions)
        self.payoff = payoff
        self.vn = vn
        self.ratio = ratio
        self.is_atk = is_atk

    def calc_value(self, prior, atk_s, dfd_s):
        prior = torch.tensor(prior, dtype=torch.float)
        a = torch.bmm(atk_s.unsqueeze(1), torch.tensor(self.payoff, dtype=torch.float))  # [t, 1, s]
        a = torch.matmul(a.squeeze(), dfd_s.unsqueeze(1))  # [t, 1]
        v = torch.matmul(prior, a.squeeze())

        if self.vn is not None:
            for a in range(self.n_actions):
                prob = torch.matmul(prior, atk_s[:, a])
                new_b = self.bayes(prior, atk_s, torch.tensor([a]))
                delta = prob * self.vn(new_b).squeeze()
                v += delta * self.ratio

        return v

    def forward(self, prior, atk_s, dfd_s, types, actions, rewards):
        # actions: [batch, actions] LongTensor
        # rewards: [batch]
        prior = ts(prior)
        new_belief = self.bayes(prior, atk_s, actions[:, 0])  # [batch, types]

        next_v = self.ratio * self.vn(new_belief) if self.vn is not None else 0.
        # if not self.is_atk:
        #     print(new_belief[0].detach().numpy(), next_v[0].detach().numpy())
        if self.is_atk:
            pi = torch.matmul(prior.unsqueeze(0), atk_s).squeeze()[actions[:, 0]]
            pi_t = atk_s[types].gather(1, actions[:, 0].unsqueeze(-1)).squeeze()
            v = rewards * torch.log(pi_t) + next_v.detach() * torch.log(pi) + next_v
        else:
            pi = dfd_s[actions[:, 1]]
            v = (rewards + next_v.detach()) * torch.log(pi) + next_v
        # print(v.detach().numpy())
        # print(rewards.detach().numpy())

        return v.mean()


def run(n_slots, n_types, prior, payoff, n_iter, plot=False, atk_vn=None, dfd_vn=None, ratio=1.0, use_pred=False):
    # _calc_exp = partial(calc_exp, n_types=n_types, n_actions=n_slots, payoff=payoff, prior=prior)
    atk_s = Strategy(n_slots, n_types)
    dfd_s = Strategy(n_slots)
    # atk_s = Strategy(n_slots, n_types, init=[[-10, 10], [-0.07740894942, -2.597107694]])
    # dfd_s = Strategy(n_slots, init=[-0.6474352071, -0.7410492519])
    atk_v = Value(n_slots, n_types, payoff[:, :, :, 0], atk_vn, True, ratio)
    dfd_v = Value(n_slots, n_types, payoff[:, :, :, 1], dfd_vn, False, ratio)

    atk_a = np.zeros((n_types, n_slots))
    dfd_a = np.zeros(n_slots)

    alr = 1e-2
    dlr = 1e-3

    st = time.time()
    avs = []
    dvs = []
    aas = []
    das = []
    aexs = [[] for _ in range(n_types)]
    dexs = []
    # print(atk_v(prior, atk_s(), dfd_s()))
    batch_size = 50
    for t in range(n_iter):
        # print(atk_s())
        # print(dfd_s())
        types = []
        actions = []
        rewards = []
        for b in range(batch_size):
            tp = np.random.choice(n_types, p=prior)
            a = np.random.choice(n_slots, p=atk_s().detach().numpy()[tp])
            b = np.random.choice(n_slots, p=dfd_s().detach().numpy())
            types.append(tp)
            actions.append([a, b])
            rewards.append(payoff[tp, a, b])
        types = torch.tensor(types)
        actions = torch.tensor(np.array(actions))
        rewards = torch.tensor(np.array(rewards))

        av = atk_v(prior, atk_s(), dfd_s(), types, actions, rewards[:, 0])
        # print(av.detach().item())
        # av = atk_v(np.ones(n_types) / n_types, atk_s(), dfd_s())
        # print(av, dv)

        decay = 1 / np.sqrt(t + 1)
        # decay = 1.
        # alr = lr / np.power(t + 1, 0.5)
        aalr = alr * decay
        adlr = dlr * decay
        # aalr = alr / np.power(t + 1, 0.5)
        # adlr = dlr / np.power(t + 1, 0.5)

        atk_s.zero_grad()
        av.backward()
        # print(atk_s.logits)
        # print(atk_s.logits.grad)
        atk_grad = atk_s.logits.grad.detach()

        # types = []
        # actions = []
        # rewards = []
        # for b in range(batch_size):
        #     tp = np.random.choice(n_types, p=prior)
        #     a = np.random.choice(n_slots, p=atk_s().detach().numpy()[tp])
        #     b = np.random.choice(n_slots, p=dfd_s().detach().numpy())
        #     types.append(tp)
        #     actions.append([a, b])
        #     rewards.append(payoff[tp, a, b])
        # types = torch.tensor(types)
        # actions = torch.tensor(np.array(actions))
        # rewards = torch.tensor(np.array(rewards))

        dv = dfd_v(prior, atk_s(), dfd_s(), types, actions, rewards[:, 1])
        # print(dv.detach().item())
        dfd_s.zero_grad()
        dv.backward()
        # print(dfd_s.logits.grad)

        dfd_s.logits.data += adlr * dfd_s.logits.grad
        dfd_a = (dfd_a * t + dfd_s().detach().numpy()) / (t + 1)

        atk_s.logits.data += aalr * atk_grad
        atk_a = (atk_a * t + atk_s().detach().numpy()) / (t + 1)

        # avs.append(atk_v(prior, ts(atk_a), ts(dfd_a)).detach().item())
        # dvs.append(dfd_v(prior, ts(atk_a), ts(dfd_a)).detach().item())
        aas.append(atk_a)
        das.append(dfd_a)

        if t % 100 == 0:
            print("#{}".format(t), atk_a, dfd_a)
            print("atk value:", atk_v.calc_value(prior, ts(atk_a), ts(dfd_a)).detach().item())
            print("dfd value:", dfd_v.calc_value(prior, ts(atk_a), ts(dfd_a)).detach().item())

    if plot:
        # iteration = list(np.arange(n_iter)) * (n_types + 1)
        # value = list(np.array(aexs).reshape(-1)) + dexs
        # name = sum([["atk-{}".format(i)] * n_iter for i in range(n_types)], []) + ["dfd"] * n_iter
        # df = pd.DataFrame(dict(iteration=iteration,
        #                        value=value,
        #                        name=name))

        df = pd.DataFrame(dict(iteration=list(np.arange(n_iter)),
                               value=np.array(das)[:, 0],
                               name=["av"] * n_iter))
        fig, ax = plt.subplots()
        # ax.set(xscale="log")
        # ax.set(yscale="log", ylim=[0.0001, 100])
        sns.lineplot(x="iteration", y="value", data=df, ax=ax, hue="name")
        plt.show()

def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.demo:
        env = DemoGame(n_round=args.n_rounds)
    else:
        env = SecurityGame(n_slots=args.n_slots, n_types=args.n_types, prior=np.array(args.prior), n_rounds=1,
                           seed=args.env_seed, random_prior=False)

    if args.n_rounds > 1:
        data = np.load("data/" + get_filename(args, args.n_rounds - 1) + ".obj")
        print(data)
        atk_vn = Interp1dPack(data[[0, 1], :].transpose())
        dfd_vn = Interp1dPack(data[[0, 2], :].transpose())
        print(atk_vn(ts([[0.5]])))
        print(dfd_vn(ts([[0.5]])))
        # exit(0)
    else:
        atk_vn = None
        dfd_vn = None

    if not args.all:
        st = time.time()
        print(run(n_slots=args.n_slots, n_types=args.n_types, prior=np.array(args.prior),
                  payoff=env.payoff, n_iter=args.n_iter, atk_vn=atk_vn, dfd_vn=dfd_vn, plot=True, ratio=args.ratio,
                  use_pred=True))
        print("Time:", time.time() - st)


if __name__ == "__main__":
    main()
