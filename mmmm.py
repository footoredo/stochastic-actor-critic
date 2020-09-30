import argparse
from env.security_game import SecurityGame
from env.demo_game import DemoGame
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from multiprocessing import Pool
from functools import partial

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from utils.utils import *
from utils.tools import load_vn
from modules.pack import Interp1dPack, CubicSplinePack
from modules.misc import Strategy, BayesFast


def parse_args():
    parser = get_parser("mmmm")
    parser.add_argument('--load-n-iter', type=int)
    parser.add_argument('--n-samples', type=int, default=20)
    parser.add_argument('--build', action="store_true", default=False)
    # parser.add_argument('--sep', action="store_true", default=False)
    parser.add_argument('--cfr', action="store_true", default=False)
    parser.add_argument('--pred', action="store_true", default=False)
    parser.add_argument('--two-nets', action="store_true", default=False)
    parser.add_argument('--verbose', action="store_true", default=False)
    return parser.parse_args()


def calc_exp(n_types, n_actions, payoff, prior, atk_s, dfd_s):
    if check_none(atk_s) or check_none(dfd_s):
        return [None] * n_types, None
    # print(payoff.shape, atk_s.shape, dfd_s.shape)
    aex = np.zeros(n_types)
    for i in range(n_types):
        tav = 0.
        tv = np.zeros(n_actions)
        for j in range(n_actions):
            tv[j] = 0.
            for k in range(n_actions):
                tv[j] += dfd_s[k] * payoff[i, j, k, 0]
            tav += tv[j] * atk_s[i, j]
        aex[i] = np.max(tv) - tav

    tv = np.zeros(n_actions)
    tav = 0.
    for k in range(n_actions):
        tv[k] = 0
        for i in range(n_types):
            for j in range(n_actions):
                tv[k] += atk_s[i, j] * prior[i] * payoff[i, j, k, 1]
        tav += tv[k] * dfd_s[k]

    dex = np.max(tv) - tav
    return aex, dex


class Value(nn.Module):
    def __init__(self, n_actions, n_types, payoff, vn=None, is_atk=False, ratio=1.0):
        super(Value, self).__init__()
        self.n_actions = n_actions
        self.n_types = n_types
        self.payoff = payoff
        self.bayes = BayesFast(n_types, n_actions)
        self.vn = vn
        self.is_atk = is_atk
        self.ratio = ratio

    def forward(self, prior, atk_s, dfd_s, eps=1e-5):
        prior = torch.tensor(prior, dtype=torch.float)
        a = torch.bmm(atk_s.unsqueeze(1), torch.tensor(self.payoff, dtype=torch.float))  # [t, 1, s]
        a = torch.matmul(a.squeeze(), dfd_s.unsqueeze(1))  # [t, 1]
        if self.is_atk:
            v = a.squeeze()
            # v = torch.matmul(prior, a.squeeze())
        else:
            v = torch.matmul(prior, a.squeeze())

        # return v
        if self.vn is not None:
            for a in range(self.n_actions):
                if self.is_atk:
                    prob = atk_s[:, a]
                    # prob = torch.matmul(prior, atk_s[:, a])
                else:
                    prob = torch.matmul(prior, atk_s[:, a])
                new_b = self.bayes(prior, atk_s, a)
                # print(prob, new_b.sum())
                # new_b = prior.unsqueeze(0)
                # v += prob * detach_ts(self.vn(new_b)) * self.ratio

                p = new_b[0].detach().item()
                aa = 0.5
                # bb = (5.0 - aa)
                bb = 10.0 - aa
                if self.is_atk:
                    tv = [0.0, 0.0]
                    if p < 0.5 - 1e-6:
                        # tv = [5.0, -5.0]
                        tv = [bb * (0.5 - p) + aa, -bb * (0.5 - p) - aa]
                    if p > 0.5 + 1e-6:
                        tv = [-bb * (p - 0.5) - aa, bb * (p - 0.5) + aa]
                else:
                    tv = aa

                if type(self.vn) == tuple or type(self.vn) == list:
                    # if self.is_atk:
                    #     print(a, prob.detach().numpy(), new_b.detach().numpy(), self.vn[0](new_b).detach().numpy(), self.vn[1](new_b).detach().numpy())
                    #     # print((prob * self.vn(new_b).detach()).detach().numpy())
                    # v += self.ratio * (prob.detach() * self.vn[0](new_b) + prob * self.vn[1](new_b).detach())

                    # print(prob.size(), ts(tv).size())
                    # xx = new_b[0] - 0.5
                    # v += self.ratio * (prob.detach() * xx * xx * 5 + prob * ts(tv))
                    v += self.ratio * (prob.detach() * self.vn[0](new_b) + prob * ts(tv))
                    # v += self.ratio * (prob.detach() * self.vn[0](new_b))
                else:
                    # if self.is_atk:
                    #     print(a, prob.detach().numpy(), new_b.detach().numpy(), self.vn(new_b).detach().numpy())
                    #     print((prob * self.vn(new_b).detach()).detach().numpy())
                    # print("erer")
                    # print(new_b.detach().numpy())
                    v += prob * self.vn(new_b, eps).detach() * self.ratio
                    # v += prob * ts(tv) * self.ratio

        return v


def pred_all(n_types, n_slots, n_iter, avs, dvs, aas, das):
    d = 0.9
    aas = np.array(aas)
    das = np.array(das)
    # pred(int(n_iter * d), n_iter, das[:, 0])
    # return None
    pav = np.array([pred(int(n_iter * d), n_iter, np.array(avs)[:, i]) for i in range(n_types)])
    # print(pav)
    # return None
    pdv = pred(int(n_iter * d), n_iter, dvs)
    # # print(aas.shape, das.shape)
    pas = [[], []]
    pds = []
    for j in range(n_slots):
        for i in range(n_types):
            pas[i].append(pred(int(n_iter * d), n_iter, aas[:, i, j]))
        pds.append(pred(int(n_iter * d), n_iter, das[:, j]))

    pas = np.array(pas)
    pds = np.array(pds)
    for i in range(n_types):
        pas[i] = regularize(pas[i])
    pds = regularize(pds)
    if check_none(pas):
        pas = aas[-1]
    if check_none(pds):
        pas = das[-1]

    return pav, pdv, pas, pds


def run_cfr(n_slots, n_types, prior, payoff, n_iter, plot=False, atk_vn=None, dfd_vn=None, ratio=1.0, use_pred=False):
    _calc_exp = partial(calc_exp, n_types=n_types, n_actions=n_slots, payoff=payoff, prior=prior)
    atk_regret = np.zeros((n_types, n_slots))
    dfd_regret = np.zeros(n_slots)
    atk_as = np.zeros((n_types, n_slots))
    dfd_as = np.zeros(n_slots)
    bayes = BayesFast(n_types, n_slots)

    def get_s(regret):
        # return np.exp(regret) / np.exp(regret).sum()
        s = np.maximum(regret, 0.)
        ss = s.sum()
        if ss < 1e-6:
            if np.min(regret) > -1e-6:
                return np.ones_like(regret) / regret.shape[-1]
                # x = np.random.rand(regret.shape[0])
                # return x / x.sum()
            else:
                x = np.argmax(regret)
                s = np.zeros_like(s)
                s[x] = 1.
                return s
        else:
            return s / ss

    atk_us = []
    dfd_us = []
    atk_ass = []
    dfd_ass = []
    atk_css = []
    dfd_css = []
    atk_es = []
    dfd_es = []

    atk_av_u = np.zeros(n_types)
    dfd_av_u = 0.

    def calc_u(a_s, d_s):
        atk_u = np.zeros((n_types, n_slots))
        atk_au = np.zeros(n_types)
        dfd_u = np.zeros(n_slots)
        dfd_au = 0.

        # print(a_s)
        for t in range(n_types):
            for a in range(n_slots):
                new_belief = bayes(prior, a_s, a)
                if any(np.isnan(new_belief)):
                    new_belief = np.ones(n_types) / n_types
                new_belief = ts(new_belief)
                for b in range(n_slots):
                    atk_r = payoff[t, a, b, 0] + (atk_vn(new_belief)[t] * ratio if atk_vn is not None else 0.)
                    dfd_r = payoff[t, a, b, 1] + (dfd_vn(new_belief) * ratio if dfd_vn is not None else 0.)
                    atk_u[t, a] += atk_r * dfd_s[b]
                    atk_au[t] += atk_r * a_s[t, a] * d_s[b]
                    dfd_u[b] += dfd_r * prior[t] * a_s[t, a]
                    dfd_au += dfd_r * prior[t] * a_s[t, a] * d_s[b]
        return atk_u, atk_au, dfd_u, dfd_au

    wsr = 0
    wss = 0
    for tt in range(n_iter):
        atk_s = []
        for t in range(n_types):
            atk_s.append(get_s(atk_regret[t]))
        atk_s = np.array(atk_s)
        dfd_s = get_s(dfd_regret)

        atk_u, atk_au, dfd_u, dfd_au = calc_u(atk_s, dfd_s)

        wr = 1
        for t in range(n_types):
            for a in range(n_slots):
                r = atk_u[t, a] - atk_au[t]
                # r += np.random.normal(0., 5.)
                # atk_regret[t, a] += r
                # print(t, a, r)
                atk_regret[t, a] = (atk_regret[t, a] * wsr + r * wr) / (wsr + wr)

        atk_s = []
        for t in range(n_types):
            atk_s.append(get_s(atk_regret[t]))
        atk_s = np.array(atk_s)
        atk_u, atk_au, dfd_u, dfd_au = calc_u(atk_s, dfd_s)

        atk_av_u = (atk_av_u * tt + atk_au) / (tt + 1)
        dfd_av_u = (dfd_av_u * tt + dfd_au) / (tt + 1)

        # print(atk_u, dfd_s)

        for b in range(n_slots):
            r = dfd_u[b] - dfd_au
            # r += np.random.normal(0., 5.)
            # dfd_regret[b] += r
            dfd_regret[b] = (dfd_regret[b] * wsr + r * wr) / (wsr + wr)

        wsr += wr

        atk_css.append(atk_s)
        dfd_css.append(dfd_s)

        ws = max(0, tt - n_iter // 10)
        # ws = 1
        if wss + ws > 0:
            atk_as = (atk_as * wss + atk_s * ws) / (wss + ws)
            dfd_as = (dfd_as * wss + dfd_s * ws) / (wss + ws)

        # atk_as = np.mean(np.array(atk_css), 0)
        # dfd_as = np.mean(np.array(dfd_css), 0)

        wss += ws

        atk_u, atk_au, dfd_u, dfd_au = calc_u(atk_as, dfd_as)

        atk_us.append(atk_au)
        dfd_us.append(dfd_au)
        atk_ass.append(atk_as)
        dfd_ass.append(dfd_as)

        # atk_e, dfd_e = _calc_exp(atk_s=atk_as, dfd_s=dfd_as)
        # atk_es.append(atk_e)
        # dfd_es.append(dfd_e)

        # if tt % 100 == 0:
            # print(atk_u, atk_au, dfd_u, dfd_au)
            # print(atk_regret, dfd_regret)

    if plot:
        fig, ax = plt.subplots()
        df = pd.DataFrame(dict(it=list(range(n_iter)) * 3,
                               u=list(np.array(atk_ass)[:, 0, 0]) + list(np.array(atk_ass)[:, 1, 0]) + list(np.array(dfd_ass)[:, 0]),
                               name=["a0"] * n_iter + ["a1"] * n_iter + ["d"] * n_iter))

        # df = pd.DataFrame(dict(it=list(range(n_iter)),
        #                        u=list(0.7712594074 - np.array(atk_css)[:, 1, 0]),
        #                        name=["a0"] * n_iter))
        # df = pd.DataFrame(dict(it=list(range(n_iter)) * 3,
        #                        u=list(np.array(atk_es)[:, 0]) + list(np.array(atk_es)[:, 1]) + dfd_es,
        #                        name=["a0"] * n_iter + ["a1"] * n_iter + ["d"] * n_iter))
        # print(atk_es[-1], dfd_es[-1])
        # df = pd.DataFrame(dict(it=list(range(n_iter)),
        #                        u=np.abs(np.array(atk_ass)[:, 0, 0] - np.array(atk_ass[:, 1, 0])),
        #                        name=["diff"] * n_iter))
        # ax.set(xscale="log")
        # ax.set(yscale="log", ylim=[0.0001, 100])
        sns.lineplot(x="it", y="u", data=df, ax=ax, hue="name")
        plt.show()

    if use_pred:
        pav, pdv, pas, pds = pred_all(n_types, n_slots, n_iter, atk_us, dfd_us, atk_ass, dfd_ass)
        # _, pav, _, pdv = calc_u(pas, pds)
        # pav = np.sum(prior * np.array(pav))
        return [pav, pdv, pas, pds, None]
    else:
        return [atk_us[-1], dfd_us[-1], atk_as, dfd_as, None]
        # return [atk_av_u, dfd_av_u, atk_as, dfd_as, None]


def run(n_slots, n_types, prior, payoff, n_iter, plot=False, atk_vn=None, dfd_vn=None, ratio=1.0, use_pred=False,
        lr=None, analyse=False):
    # if solve:
    #     return np.ones((n_types, n_slots)) / n_slots, np.ones(n_slots) / n_slots
    _calc_exp = partial(calc_exp, n_types=n_types, n_actions=n_slots, payoff=payoff, prior=prior)
    # atk_s = Strategy(n_slots, n_types, init=np.random.normal(0., .1, (n_types, n_slots)))
    # dfd_s = Strategy(n_slots, init=np.random.normal(0., .1, n_slots))
    atk_s = Strategy(n_slots, n_types)
    dfd_s = Strategy(n_slots)
    # atk_s = Strategy(n_slots, n_types, init=[[-10, 10], [-0.07740894942, -2.597107694]])
    # dfd_s = Strategy(n_slots, init=[-0.6474352071, -0.7410492519])
    atk_v = Value(n_slots, n_types, payoff[:, :, :, 0], atk_vn, True, ratio)
    dfd_v = Value(n_slots, n_types, payoff[:, :, :, 1], dfd_vn, False, ratio)

    atk_a = np.zeros((n_types, n_slots))
    dfd_a = np.zeros(n_slots)

    alr = (lr or .1) * 1
    dlr = lr or .1

    st = time.time()
    avs = []
    dvs = []
    aas = []
    das = []
    acs = []
    dcs = []
    aexs = [[] for _ in range(n_types)]
    dexs = []
    ws = 0
    # print(atk_v(prior, atk_s(), dfd_s()))
    # start_t = n_iter / 4
    start_t = 0
    # eps_k = np.log(1.) / n_iter
    # start_eps = 0
    eps_k = np.log(1000.) / n_iter
    start_eps = 1e-2
    for t in range(n_iter):
        # print(atk_s())
        # print(dfd_s())
        eps = start_eps / np.exp(eps_k * t)
        # eps = start_eps
        av = atk_v(prior, atk_s(), dfd_s(), ts(eps))
        # av = atk_v(np.ones(n_types) / n_types, atk_s(), dfd_s())
        # print(av, dv)

        # alr = lr / np.power(t + 1, 0.5)
        # decay = 1. / np.sqrt(t + 1)
        decay = 1.0
        aalr = alr * decay
        adlr = dlr * decay
        # aalr = alr / np.power(t + 1, 0.5)
        # adlr = dlr / np.power(t + 1, 0.5)

        atk_s.zero_grad()
        av.sum().backward()
        # (av * ts(prior)).sum().backward()
        # print(atk_s.logits)
        # print(atk_s.logits.grad)
        atk_grad = atk_s.logits.grad.detach().numpy().copy()
        # if (t + 1) % 100 == 0:
        #     print(atk_grad)
        with torch.no_grad():
            atk_s.logits += aalr * (atk_s.logits.grad + eps * torch.normal(torch.zeros_like(atk_s.logits), 1.))
            atk_s.logits.grad.zero_()

        w = 1 if t >= start_t else 0
        if ws + w > 0:
            atk_a = (atk_a * ws + atk_s().detach().numpy() * w) / (ws + w)
        else:
            atk_a = atk_s().detach().numpy()

        dv = dfd_v(prior, atk_s(), dfd_s())
        dfd_s.zero_grad()
        dv.backward()
        # print(dfd_s.logits.grad)
        dfd_grad = dfd_s.logits.grad.detach().numpy().copy()
        with torch.no_grad():
            dfd_s.logits += adlr * (dfd_s.logits.grad + eps * torch.normal(torch.zeros_like(dfd_s.logits), 1.))
            dfd_s.logits.grad.zero_()

        if ws + w > 0:
            dfd_a = (dfd_a * ws + dfd_s().detach().numpy() * w) / (ws + w)
        else:
            dfd_a = dfd_s().detach().numpy()

        ws += w

        avs.append(atk_v(prior, ts(atk_a), ts(dfd_a)).detach().numpy())
        dvs.append(dfd_v(prior, ts(atk_a), ts(dfd_a)).detach().item())
        aas.append(atk_a)
        das.append(dfd_a)
        acs.append(atk_s().detach().numpy())
        dcs.append(dfd_s().detach().numpy())
        aex, dex = _calc_exp(atk_s=atk_a, dfd_s=dfd_a)
        for i in range(n_types):
            aexs[i].append(aex[i])
        dexs.append(dex)

        # if (t + 1) % 100 == 0:
        #     print("#", t)
        #     # print("atk_grad_norm:", np.linalg.norm(atk_grad))
        #     # print("dfd_grad_norm:", np.linalg.norm(dfd_grad))
        #     print("atk_grad:", atk_grad)
        #     print("dfd_grad:", dfd_grad)
        #     print("atk_logits:", atk_s.logits.data)
        #     print("dfd_logits:", dfd_s.logits.data)
        #     print(atk_s().detach().numpy())
        #     print(dfd_s().detach().numpy())
        #
        #     print(atk_v(prior, atk_s(), dfd_s()))
        #     print(dfd_v(prior, atk_s(), dfd_s()))

    # print(time.time() - st)
    #

    if analyse:
        start = 500
        return analyse_wave(list(np.array(acs)[start:, 0, 0])), analyse_wave(list(np.array(acs)[start:, 1, 0])), \
               analyse_wave(list(np.array(dcs)[start:, 0]))

    if plot:
        # analyse_wave(list(np.array(acs)[:, 1, 0]))
        # iteration = list(np.arange(n_iter)) * (n_types + 1)
        # value = list(np.array(aexs).reshape(-1)) + dexs
        # name = sum([["atk-{}".format(i)] * n_iter for i in range(n_types)], []) + ["dfd"] * n_iter
        # iteration = list(np.arange(n_iter))
        # value = np.array(aexs[0]) + np.array(aexs[1]) + np.array(dexs)
        # name = ["sum"] * n_iter
        # df = pd.DataFrame(dict(iteration=iteration,
        #                        value=value,
        #                        name=name))

        # np.array(aas).dump("two-nets.data")
        # np.array(aas).dump("one-net.data")
        # df = pd.DataFrame(dict(iteration=list(range(n_iter)),
        #                        value=np.abs(np.array(aas)[:, 0, 0] - np.array(aas)[:, 1, 0]),
        #                        name=["diff"] * n_iter))
        df = pd.DataFrame(dict(iteration=list(np.arange(n_iter)) + list(np.arange(n_iter)) + list(np.arange(n_iter)),
                               value=list(np.array(acs)[:, 0, 0]) + list(np.array(acs)[:, 1, 0]) + list(
                                   np.array(dcs)[:, 0]),
                               name=["a0"] * n_iter + ["a1"] * n_iter + ["d"] * n_iter))
        # df = pd.DataFrame(dict(iteration=list(range(4000, n_iter)),
        #                        value=list(np.array(dcs)[4000:, 0]),
        #                        name=["d"] * (n_iter - 4000)))
        # df = pd.DataFrame(dict(iteration=list(np.arange(n_iter)),
        #                        value=list(np.abs(np.array(aas)[:, 1, 0] - 0.5141729383)),
        #                        # value=list(np.array(aas)[:, 0, 0] - 0.1045),
        #                        name=["a0"] * n_iter))
        fig, ax = plt.subplots()
        # ax.set(xscale="log")
        # ax.set(yscale="log", ylim=[0.0000001, 100])
        # ax.set(ylim=[0.0, 1.0])
        sns.lineplot(x="iteration", y="value", data=df, ax=ax, hue="name")
        plt.show()

    if use_pred:
        # pred(int(n_iter * 0.9), n_iter, np.array(dcs)[:, 0])
        # return None
        pav, pdv, pas, pds = pred_all(n_types, n_slots, n_iter, avs, dvs, aas, das)
        # print(pav, pdv)
        # pav = (atk_v(prior, ts(pas), ts(pds))).detach().numpy()
        # pdv = (dfd_v(prior, ts(pas), ts(pds))).detach().item()
        # print(pav, pdv)
        return [pav, pdv, pas, pds, lr]
    else:
        return [avs[-1], dvs[-1], atk_a, dfd_a, lr]


class Runner(object):
    def __init__(self, args, payoff, atk_vn, dfd_vn, n_iter=None, use_pred=False, plot=False, analyse=False):
        self.args = args
        self.payoff = payoff
        self.atk_vn = atk_vn
        self.dfd_vn = dfd_vn
        self.use_pred = use_pred
        self.plot = plot
        self.analyse = analyse
        self.n_iter = n_iter or args.n_iter

    def __call__(self, x):
        args = self.args
        if type(x) == tuple:
            p, lr, it_k = x
        else:
            p = x
            lr = None
            it_k = 1
        return run(n_slots=args.n_slots, n_types=args.n_types, prior=np.array([p, 1 - p]),
                   payoff=self.payoff, n_iter=self.n_iter * it_k, atk_vn=self.atk_vn, dfd_vn=self.dfd_vn, plot=self.plot,
                   ratio=args.ratio, use_pred=self.use_pred, analyse=self.analyse, lr=lr)


class CFRRunner(object):
    def __init__(self, args, payoff, atk_vn, dfd_vn, plot=False, use_pred=False):
        self.args = args
        self.payoff = payoff
        self.atk_vn = atk_vn
        self.dfd_vn = dfd_vn
        self.plot = plot
        self.use_pred = use_pred

    def __call__(self, p):
        args = self.args
        return run_cfr(n_slots=args.n_slots, n_types=args.n_types, prior=np.array([p, 1 - p]),
                       payoff=self.payoff, n_iter=args.n_iter, atk_vn=self.atk_vn, dfd_vn=self.dfd_vn,
                       ratio=args.ratio, plot=self.plot, use_pred=self.use_pred)


def test_lr(runner, p, lr_list):
    def _approve(res):
        # print(res)
        if res is None:
            return 2  # close to 0 or 1
        n, hd, ha = res
        # if hd > 1e-3:
        #     return False
        if hd > 2e-6:
            return 0
        if ha > 0.5:
            return 0
        if n < 10:
            return 0
        if hd < -5e-6:
            return 1
        if n > 40:
            return 1
        return 2

    def approve(res):
        return _approve(res) == 2

    def semi_approve(res):
        return _approve(res) > 0

    results = []
    for lr in reversed(sorted(lr_list)):
        print("Testing", lr)
        a0, a1, d = runner((p, lr, 1))
        print(p, lr, a0, a1, d)
        none_count = 0
        if a0 is None:
            none_count += 1
        if a1 is None:
            none_count += 1
        if d is None:
            none_count += 1
        # if none_count > 1:
        #     continue
        # if a0 is None and a1 is None and d is None:
        #     continue
        if approve(a0) and approve(a1) and approve(d):
            return lr, a0, a1, d
        if semi_approve(a0) and semi_approve(a1) and semi_approve(d):
            results.append((lr, a0, a1, d))

    min_dis = 1.
    min_i = None

    def update_dis(dis, r):
        if r is None:
            return dis
        n, hd, ha = r
        if ha > 0.5:
            return 100.
        return max(dis, abs(hd))

    for i, x in enumerate(results):
        lr, a0, a1, d = x
        cur_dis = 0.
        cur_dis = update_dis(cur_dis, a0)
        cur_dis = update_dis(cur_dis, a1)
        cur_dis = update_dis(cur_dis, d)
        if cur_dis < min_dis:
            min_dis = cur_dis
            min_i = i
    return None if min_i is None else results[min_i]


class RunnerWithTest(object):
    def __init__(self, args, payoff, atk_vn, dfd_vn, n_iter=None, use_pred=False, plot=False):
        self.test_runner = Runner(args, payoff, atk_vn, dfd_vn, n_iter=2000, use_pred=False, plot=False, analyse=True)
        self.runner = Runner(args, payoff, atk_vn, dfd_vn, n_iter=n_iter, use_pred=use_pred, plot=plot, analyse=False)

    def __call__(self, p):
        # lr = 0.552061438912436
        # lr = 0.5
        # lr = 0.7
        # lr = test_lr(self.test_runner, p, [0.01 * (1.2 ** i) for i in range(26)])
        # lr, a0, a1, d = test_lr(self.test_runner, p, np.arange(0.05, 1.0, 0.05))
        if True:
            # lr = 0.07378945279999996
            lr = 0.01
        else:
            ret = test_lr(self.test_runner, p, [0.005 * (1.4 ** i) for i in range(15)])
            # ret = test_lr(self.test_runner, p, np.arange(0.4, 0.5, 0.01))
            if ret is None:
                print("lr failed at", p)
                lr = 0.005
            else:
                lr, a0, a1, d = ret
        # it_k = 5
        # if a0 is not None and a0[0] > 10:
        #     it_k = 1
        # if a1 is not None and a1[0] > 10:
        #     it_k = 1
        # if d is not None and d[0] > 10:
        #     it_k = 1
        # lr = 0.01
        # test_lr(self.test_runner, p, [lr])
        # lr = 0.2024782584831999
        it_k = 1
        # print(p, "pick", lr)
        return self.runner((p, lr, it_k))


class RunnerWithSn(object):
    def __init__(self, n_types, n_slots, atk_sn, dfd_sn):
        self.n_types = n_types
        self.n_slots = n_slots
        self.atk_sn = atk_sn
        self.dfd_sn = dfd_sn

    def __call__(self, p):
        b = ts([p])
        return None, None, to_np(self.atk_sn(b)).reshape(self.n_types, self.n_slots), to_np(self.dfd_sn(b)), None


def build_strategy(args, n_slots, n_types, n_rounds, ps, payoff, n_iter, atk_vns, dfd_vns, atk_sns=None, dfd_sns=None,
                   ratio=1.0):
    assert len(atk_vns) == n_rounds - 1
    assert len(dfd_vns) == n_rounds - 1

    def history_encode(h):
        code = 0
        b = 1
        for a in h:
            code += b * a
            b *= n_slots
        return code + (n_slots ** len(h) - 1)

    n = len(ps)
    atk_s = np.zeros((n, n_slots ** n_rounds - 1, n_types, n_slots))
    dfd_s = np.zeros((n, n_slots ** n_rounds - 1, n_slots))
    bayes = BayesFast(n_types, n_slots)

    # def solve(h, belief):
    #     remain = n_rounds - len(h) - 1
    #     atk_vn = None if remain == 0 else atk_vns[remain - 1]
    #     dfd_vn = None if remain == 0 else dfd_vns[remain - 1]
    #     atk_ts, dfd_ts = run(n_slots, n_types, prior=belief, payoff=payoff, n_iter=n_iter, atk_vn=atk_vn,
    #                          dfd_vn=dfd_vn, ratio=ratio, plot=False, use_pred=False, solve=True)
    #     eh = history_encode(h)
    #     # print(eh, h)
    #     atk_s[eh, :, :] = np.array(atk_ts)
    #     dfd_s[eh, :] = np.array(dfd_ts)
    #
    #     if remain > 0:
    #         for a in range(n_slots):
    #             new_belief = bayes(belief, ts(atk_ts), a)
    #             solve(h + [a], new_belief)

    # solve([], ts(prior))

    atk_avs = []
    dfd_avs = []
    atk_tss = []
    dfd_tss = []
    lrs = []

    def width_first_solve():
        queue = [(i, [], p) for i, p in enumerate(ps)]
        for i in range(n_rounds):
            next_queue = []
            remain = n_rounds - i - 1
            atk_vn = None if remain == 0 else atk_vns[remain - 1]
            dfd_vn = None if remain == 0 else dfd_vns[remain - 1]
            if args.verbose:
                print("Depth: {}, with {} instance(s) to run.".format(i, len(queue)))
                print([x[2] for x in queue])
            if False:
            # if i > 0 and atk_sns is not None and dfd_sns is not None:
                atk_sn = atk_sns[remain]
                dfd_sn = dfd_sns[remain]
                runner = RunnerWithSn(args.n_types, args.n_slots, atk_sn, dfd_sn)
            else:
                if args.cfr:
                    runner = CFRRunner(args, payoff, atk_vn, dfd_vn, use_pred=args.pred)
                else:
                    runner = RunnerWithTest(args, payoff, atk_vn, dfd_vn, use_pred=args.pred)
            # print(i, [x[1] for x in queue])
            with Pool(12) as p:
                tss = list(p.map(runner, [x[2] for x in queue]))
                for j, x in enumerate(queue):
                    idx, h, p = x
                    belief = np.array([p, 1 - p])
                    eh = history_encode(h)
                    atk_av, dfd_av, atk_ts, dfd_ts, lr = tss[j]
                    # print(x, atk_ts, dfd_ts)
                    if i == 0:
                        atk_avs.append(atk_av)
                        dfd_avs.append(dfd_av)
                        atk_tss.append(atk_ts)
                        dfd_tss.append(dfd_ts)
                        lrs.append(lr)
                    atk_s[idx, eh, :, :] = np.array(atk_ts)
                    dfd_s[idx, eh, :] = np.array(dfd_ts)

                    if remain > 0:
                        for a in range(n_slots):
                            new_belief = bayes(ts(belief), ts(atk_ts), a).detach().numpy()
                            # print(new_belief)
                            next_queue.append((idx, h + [a], new_belief[0]))
            queue = next_queue

    width_first_solve()

    def atk_fn(i, t, h):
        return atk_s[i, history_encode(h), t]

    def dfd_fn(i, h):
        return dfd_s[i, history_encode(h)]

    return [(partial(atk_fn, i), partial(dfd_fn, i)) for i in range(n)], atk_avs, dfd_avs, atk_tss, dfd_tss, lrs


def main():
    args = parse_args()

    if args.seed is None:
        import random
        args.seed = random.randint(1, 10000)
        if args.verbose:
            print("seed:", args.seed)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if args.verbose:
            print("Seeding with {}".format(args.seed))

    if args.demo:
        env = DemoGame(n_round=args.n_rounds)
    else:
        prior = None if args.prior is None else np.array(args.prior)
        env = SecurityGame(n_slots=args.n_slots, n_types=args.n_types, prior=prior,
                           n_rounds=args.n_rounds, seed=args.env_seed, random_prior=False)

    def _load_vn(_round):
        _name = "data/" + get_filename(args, n_rounds=_round, n_iter=args.load_n_iter or args.n_iter) + ".{}.obj"
        _interp = ["nn", "linear_fast"] if args.two_nets else "linear_fast"
        _atk_vn = load_vn(_name.format("atk_vn"), _interp)
        _dfd_vn = load_vn(_name.format("dfd_vn"), _interp)
        _atk_sn = load_vn(_name.format("atk_sn"), _interp)
        _dfd_sn = load_vn(_name.format("dfd_sn"), _interp)
        return _atk_vn, _dfd_vn, _atk_sn, _dfd_sn

    if args.n_rounds > 1 and not args.build:
        atk_vn, dfd_vn, _, _ = _load_vn(args.n_rounds - 1)
        # exit(0)
    else:
        atk_vn = None
        dfd_vn = None

    if args.all:
        ps = []

        n_samples = args.n_samples
        for i in range(n_samples):
            # p = np.random.random()
            p = i / n_samples
            ps.append(p)
        ps.append(1.0)
    else:
        n_samples = 1
        ps = [args.prior[0]]

    if args.build:
        st = time.time()
        atk_vns = []
        dfd_vns = []
        atk_sns = []
        dfd_sns = []
        for i in range(args.n_rounds - 1):
            atk_vn, dfd_vn, atk_sn, dfd_sn = _load_vn(i + 1)
            atk_vns.append(atk_vn)
            dfd_vns.append(dfd_vn)
            atk_sns.append(atk_sn)
            dfd_sns.append(dfd_sn)
        s_fns, avs, dvs, atk_tss, dfd_tss, lrs = build_strategy(args=args, n_slots=args.n_slots, n_types=args.n_types,
                                                                n_rounds=args.n_rounds,
                                                                payoff=env.payoff, ps=ps, n_iter=args.n_iter,
                                                                atk_vns=atk_vns, dfd_vns=dfd_vns,
                                                                atk_sns=atk_sns, dfd_sns=dfd_sns, ratio=args.ratio)
        if args.verbose:
            print("Time:", time.time() - st)

        atk_epss = []
        atk_pbne_epss = []
        dfd_epss = []
        dfd_pbne_epss = []
        for i, p in enumerate(ps):
            atk_fn, dfd_fn = s_fns[i]
            if args.verbose:
                print("\n========================")
                print("Assessment for {}:".format(p))
            env.prior = np.array([p, 1. - p])
            res, tss = env.assess_strategies((atk_fn, dfd_fn), verbose=args.verbose)
            if args.verbose:
                print("========================\n")
            atk_res, dfd_res = res
            atk_eps, atk_pbne_eps = atk_res
            dfd_eps, dfd_pbne_eps = dfd_res
            atk_epss.append(atk_eps)
            atk_pbne_epss.append(atk_pbne_eps)
            dfd_epss.append(dfd_eps)
            dfd_pbne_epss.append(dfd_pbne_eps)

        if args.all:
            lines = []
            formatter = "\t".join(["{}"] * 11) + "\n"
            lines.append(formatter.format(args.env_seed, "Strategy", "", "", "Distance", "", "", "PBNE Distance", "", "", ""))
            lines.append(formatter.format("n_rounds={}".format(args.n_rounds), "Attacker-0", "Attacker-1",
                                          "Defender", "Attacker-0", "Attacker-1", "Defender", "Attacker-0",
                                          "Attacker-1", "Defender", "lr"))
            for i, p in enumerate(ps):
                lines.append(formatter.format(p, atk_tss[i][0][0], atk_tss[i][1][0], dfd_tss[i][0],
                                              atk_epss[i][0], atk_epss[i][1], dfd_epss[i],
                                              atk_pbne_epss[i][0], atk_pbne_epss[i][1], dfd_pbne_epss[i], lrs[i]))
            with open("results/{}.txt".format(get_filename(args)), "w") as f:
                f.writelines(lines)

    if not args.all:
        if not args.build:
            st = time.time()
            if args.cfr:
                st = time.time()
                runner = CFRRunner(args, env.payoff, atk_vn=atk_vn, dfd_vn=dfd_vn, plot=True, use_pred=args.pred)
                print(runner(args.prior[0]))
                print("Time:", time.time() - st)

            else:
                runner = RunnerWithTest(args, env.payoff, atk_vn=atk_vn, dfd_vn=dfd_vn, plot=True, use_pred=args.pred)
                print(runner(args.prior[0]))

                # print(run(n_slots=args.n_slots, n_types=args.n_types, prior=np.array(args.prior),
                #           payoff=env.payoff, n_iter=args.n_iter, atk_vn=atk_vn, dfd_vn=dfd_vn, plot=True, ratio=args.ratio,
                #           use_pred=args.pred))
                print("Time:", time.time() - st)
    else:
        if not args.build:
            st = time.time()
            with Pool(12) as p:
                if not args.cfr:
                    vs = list(p.map(RunnerWithTest(args, payoff=env.payoff, atk_vn=atk_vn, dfd_vn=dfd_vn, use_pred=args.pred), ps))
                else:
                    vs = list(p.map(CFRRunner(args, payoff=env.payoff, atk_vn=atk_vn, dfd_vn=dfd_vn, use_pred=args.pred), ps))
                avs, dvs, atk_tss, dfd_tss, _ = map(list, zip(*vs))
                # for i, p in enumerate(ps):
                #     print("\t".join(["{}"] * 7).format(p, ex[i][0][0], ex[i][0][1], ex[i][1], exp[i][0][0], exp[i][0][1],
                #                                        exp[i][1]))

            print("Time:", time.time() - st)

        if args.verbose:
            print(avs)
            print(dvs)

            avs = np.array(avs)
            print(avs.transpose().reshape(-1))

            df = pd.DataFrame(dict(p=ps + ps, v=list(avs.transpose().reshape(-1)),
                                   t=["a"] * (n_samples + 1) + ["b"] * (n_samples + 1)))
            sns.lineplot(x="p", y="v", data=df, hue="t")
            if args.save_plot:
                plt.savefig("plot/" + get_filename(args) + ".png", dpi=100)
            else:
                plt.show()

        if args.save_data:
            n = n_samples + 1
            ps = np.array(ps).reshape(n, 1)
            avs = np.array(avs).reshape(n, env.n_types)
            dvs = np.array(dvs).reshape(n, 1)
            atk_tss = np.array(atk_tss).reshape(n, env.n_types * env.n_slots)
            dfd_tss = np.array(dfd_tss).reshape(n, env.n_slots)
            name = "data/" + get_filename(args) + ".{}.obj"
            np.concatenate([ps, avs], axis=1).dump(name.format("atk_vn"))
            np.concatenate([ps, dvs.reshape(n, 1)], axis=1).dump(name.format("dfd_vn"))
            np.concatenate([ps, atk_tss], axis=1).dump(name.format("atk_sn"))
            np.concatenate([ps, dfd_tss], axis=1).dump(name.format("dfd_sn"))


if __name__ == "__main__":
    main()
