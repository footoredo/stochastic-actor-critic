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
    parser.add_argument('--build', action="store_true", default=False)
    return parser.parse_args()


sss = """-0.5289521415	1.92078265	1.47132343
-0.6000796998	1.927774621	1.470627338
-0.6910867266	1.955017832	1.471201841
-0.8163690535	2.018780501	1.477761271
-1.281290202	2.255433542	1.477046199
-1.345538948	2.295303546	1.471998621
-1.360456279	2.326823054	1.482942076
-1.424982532	2.444170809	1.473495646
-1.556015033	2.948649852	1.477981844
-1.60378908	3.10208818	1.491053313
-1.619093747	3.295927726	1.513530379"""

# sss = """-0.3938881987	1.921428326	1.921428326	1.471743537
# -0.3938881987	1.921428326	1.689896673	1.471743537
# -0.3938881987	1.921428326	1.458365021	1.471743537
# -1.330031419	2.331433031	1.232993696	1.471743537
# -1.330031419	2.331433031	0.8668472511	1.471743537
# -1.330031419	2.331433031	0.5007008062	1.471743537
# -1.34582891	2.351454858	0.1330845976	1.471743537
# -1.34582891	2.351454858	-0.2366437792	1.471743537
# -1.619176196	3.474365121	-0.6004679326	1.471743537
# -1.619176196	3.474365121	-1.109822064	1.471743537
# -1.619176196	3.474365121	-1.619176196	1.471743537"""

# sss = np.array(list(map(float, sss.split()))).reshape((11, 3))
# # atk_vn = []
# samples = []
# for p in range(11):
#     s = p / 10. * sss[p, 0] + (10 - p) / 10. * sss[p, 1]
#     samples.append((np.array([p / 10, (10 - p) / 10]), np.array([s])))
#     # print(s)
#     # atk_vn.append(Pack(samples))
# atk_vn = CubicSplinePack(samples)
# # print(atk_vn(ts([0.5])))
#
# samples = []
# for p in range(11):
#     samples.append((np.array([p / 10, (10 - p) / 10]), sss[p, 2].reshape(1)))
# dfd_vn = CubicSplinePack(samples)


def check_none(a):
    if a is None:
        return True
    for x in list(a.reshape(-1)):
        if x is None:
            return True
    return False


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

    def forward(self, prior, atk_s, dfd_s):
        prior = torch.tensor(prior, dtype=torch.float)
        a = torch.bmm(atk_s.unsqueeze(1), torch.tensor(self.payoff, dtype=torch.float))  # [t, 1, s]
        a = torch.matmul(a.squeeze(), dfd_s.unsqueeze(1))  # [t, 1]
        v = torch.matmul(prior, a.squeeze())

        if self.vn is not None:
            for a in range(self.n_actions):
                prob = torch.matmul(prior, atk_s[:, a])
                new_b = self.bayes(prior, atk_s, a)
                # print(prob, new_b.sum())
                # new_b = prior.unsqueeze(0)
                v += prob * self.vn(new_b) * self.ratio

        return v


def regularize(a):
    if check_none(a):
        return None
    a = a.clip(0., 1.)
    return a / np.sum(a)


def run(n_slots, n_types, prior, payoff, n_iter, plot=False, atk_vn=None, dfd_vn=None, ratio=1.0, use_pred=False,
        solve=False):
    # if solve:
    #     return np.ones((n_types, n_slots)) / n_slots, np.ones(n_slots) / n_slots
    _calc_exp = partial(calc_exp, n_types=n_types, n_actions=n_slots, payoff=payoff, prior=prior)
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
    for t in range(n_iter):
        # print(atk_s())
        # print(dfd_s())
        av = atk_v(prior, atk_s(), dfd_s())
        # av = atk_v(np.ones(n_types) / n_types, atk_s(), dfd_s())
        # print(av, dv)

        # alr = lr / np.power(t + 1, 0.5)
        # decay = 1 / np.sqrt(t + 1)
        decay = 1.0
        aalr = alr * decay
        adlr = dlr * decay
        # aalr = alr / np.power(t + 1, 0.5)
        # adlr = dlr / np.power(t + 1, 0.5)

        atk_s.zero_grad()
        av.backward()
        # print(atk_s.logits)
        # print(atk_s.logits.grad)
        atk_s.logits.data += aalr * atk_s.logits.grad
        atk_a = (atk_a * t + atk_s().detach().numpy()) / (t + 1)

        dv = dfd_v(prior, atk_s(), dfd_s())
        dfd_s.zero_grad()
        dv.backward()
        # print(dfd_s.logits.grad)
        dfd_s.logits.data += adlr * dfd_s.logits.grad

        dfd_a = (dfd_a * t + dfd_s().detach().numpy()) / (t + 1)

        avs.append(atk_v(prior, ts(atk_a), ts(dfd_a)).detach().item())
        dvs.append(dfd_v(prior, ts(atk_a), ts(dfd_a)).detach().item())
        aas.append(atk_a)
        das.append(dfd_a)
        # aex, dex = _calc_exp(atk_s=atk_a, dfd_s=dfd_a)
        # for i in range(n_types):
        #     aexs[i].append(aex[i])
        # dexs.append(dex)

        # if (t + 1) % 100 == 0:
        #     print(atk_a)
        #     print(dfd_a)
        #
        #     print(atk_v(prior, ts(atk_a), ts(dfd_a)))
        #     print(dfd_v(prior, ts(atk_a), ts(dfd_a)))

    # print(time.time() - st)
    #

    if plot:
        # iteration = list(np.arange(n_iter)) * (n_types + 1)
        # value = list(np.array(aexs).reshape(-1)) + dexs
        # name = sum([["atk-{}".format(i)] * n_iter for i in range(n_types)], []) + ["dfd"] * n_iter
        # df = pd.DataFrame(dict(iteration=iteration,
        #                        value=value,
        #                        name=name))

        df = pd.DataFrame(dict(iteration=list(np.arange(n_iter)),
                               value=np.abs(-0.2366437792 - np.array(avs)),
                               name=["av"] * n_iter))
        fig, ax = plt.subplots()
        ax.set(xscale="log")
        ax.set(yscale="log", ylim=[0.0001, 100])
        sns.lineplot(x="iteration", y="value", data=df, ax=ax, hue="name")
        plt.show()

    pav = None
    pdv = None
    pas = None
    pds = None
    d = 0.25
    if use_pred:
        pav = pred(int(n_iter * d), n_iter, avs)
        # print(pav)
        # return None
        pdv = pred(int(n_iter * d), n_iter, dvs)
        aas = np.array(aas)
        das = np.array(das)
        # print(aas.shape, das.shape)
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
            pas = atk_a
        if check_none(pds):
            pas = dfd_a
        # print("without prediction:", avs[-1], dvs[-1], atk_a, dfd_a)
        # print("prediction:", pav, pdv, pas, pds)
        # print("value with predicted action:", atk_v(prior, ts(pas), ts(pds)).detach().item(),
        #       dfd_v(prior, ts(pas), ts(pds)).detach().item())
        # print("exploitability:", calc_exp(n_types, n_slots, payoff, prior, atk_a, dfd_a))
        # print("exploitability with predicted action:", calc_exp(n_types, n_slots, payoff, prior, pas, pds))

    # print(atk_a, dfd_a)
    # return [atk_v(prior, ts(atk_a), ts(dfd_a)).detach().item(), dfd_v(prior, ts(atk_a), ts(dfd_a)).detach().item()]
    if solve:
        if use_pred:
            return pas, pds
        else:
            return atk_a, dfd_a
    if use_pred:
        # pav = None
        # pdv = None
        return [pav or avs[-1], pdv or dvs[-1], calc_exp(n_types, n_slots, payoff, prior, atk_a, dfd_a),
                calc_exp(n_types, n_slots, payoff, prior, pas, pds)]
    else:
        return [avs[-1], dvs[-1], None, None]


class Runner(object):
    def __init__(self, args, payoff, atk_vn, dfd_vn, solve=False):
        self.args = args
        self.payoff = payoff
        self.atk_vn = atk_vn
        self.dfd_vn = dfd_vn
        self.solve = solve

    def __call__(self, x):
        args = self.args
        return run(n_slots=args.n_slots, n_types=args.n_types, prior=np.array([x, 1 - x]),
                   payoff=self.payoff, n_iter=args.n_iter, atk_vn=self.atk_vn, dfd_vn=self.dfd_vn, plot=False,
                   ratio=args.ratio, use_pred=False, solve=self.solve)


def build_strategy(args, n_slots, n_types, n_rounds, prior, payoff, n_iter, atk_vns, dfd_vns, ratio=1.0):
    assert len(atk_vns) == n_rounds - 1
    assert len(dfd_vns) == n_rounds - 1

    def history_encode(h):
        code = 0
        b = 1
        for a in h:
            code += b * a
            b *= n_slots
        return code + (n_slots ** len(h) - 1)

    atk_s = np.zeros((n_slots ** n_rounds - 1, n_types, n_slots))
    dfd_s = np.zeros((n_slots ** n_rounds - 1, n_slots))
    bayes = BayesFast(n_types, n_slots)

    def solve(h, belief):
        remain = n_rounds - len(h) - 1
        atk_vn = None if remain == 0 else atk_vns[remain - 1]
        dfd_vn = None if remain == 0 else dfd_vns[remain - 1]
        atk_ts, dfd_ts = run(n_slots, n_types, prior=belief, payoff=payoff, n_iter=n_iter, atk_vn=atk_vn,
                             dfd_vn=dfd_vn, ratio=ratio, plot=False, use_pred=True, solve=True)
        eh = history_encode(h)
        # print(eh, h)
        atk_s[eh, :, :] = np.array(atk_ts)
        dfd_s[eh, :] = np.array(dfd_ts)

        if remain > 0:
            for a in range(n_slots):
                new_belief = bayes(belief, ts(atk_ts), a)
                solve(h + [a], new_belief)

    # solve([], ts(prior))

    def width_first_solve():
        queue = [([], prior)]
        for i in range(n_rounds):
            next_queue = []
            remain = n_rounds - i - 1
            atk_vn = None if remain == 0 else atk_vns[remain - 1]
            dfd_vn = None if remain == 0 else dfd_vns[remain - 1]
            print(i, remain - 1)
            runner = Runner(args, payoff, atk_vn, dfd_vn, True)
            # print(i, [x[1] for x in queue])
            with Pool(12) as p:
                tss = list(p.map(runner, [x[1][0] for x in queue]))
                for j, x in enumerate(queue):
                    h, belief = x
                    eh = history_encode(h)
                    atk_ts, dfd_ts = tss[j]
                    atk_s[eh, :, :] = np.array(atk_ts)
                    dfd_s[eh, :] = np.array(dfd_ts)

                    if remain > 0:
                        for a in range(n_slots):
                            new_belief = bayes(ts(belief), ts(atk_ts), a)
                            next_queue.append((h + [a], new_belief.detach().numpy()))
            queue = next_queue

    width_first_solve()

    def atk_fn(t, h):
        return atk_s[history_encode(h), t]

    def dfd_fn(h):
        return dfd_s[history_encode(h)]

    return atk_fn, dfd_fn


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.demo:
        env = DemoGame(n_round=args.n_rounds)
    else:
        env = SecurityGame(n_slots=args.n_slots, n_types=args.n_types, prior=np.array(args.prior),
                           n_rounds=args.n_rounds, seed=args.env_seed, random_prior=False)

    if args.n_rounds > 1 and not args.build:
        atk_vn, dfd_vn = load_vn("data/" + get_filename(args, n_rounds=args.n_rounds - 1,
                                                        n_iter=args.load_n_iter) + ".obj")
        # exit(0)
    else:
        atk_vn = None
        dfd_vn = None

    if args.build:
        st = time.time()
        atk_vns = []
        dfd_vns = []
        for i in range(args.n_rounds - 1):
            atk_vn, dfd_vn = load_vn("data/" + get_filename(args, n_rounds=i + 1, n_iter=args.load_n_iter) + ".obj")
            atk_vns.append(atk_vn)
            dfd_vns.append(dfd_vn)
        atk_fn, dfd_fn = build_strategy(args=args, n_slots=args.n_slots, n_types=args.n_types, n_rounds=args.n_rounds,
                                        payoff=env.payoff, prior=np.array(args.prior), n_iter=args.n_iter,
                                        atk_vns=atk_vns, dfd_vns=dfd_vns, ratio=args.ratio)
        print("Time:", time.time() - st)
        print("Assessment:", env.assess_strategies((atk_fn, dfd_fn)))
    elif not args.all:
        st = time.time()
        print(run(n_slots=args.n_slots, n_types=args.n_types, prior=np.array(args.prior),
                  payoff=env.payoff, n_iter=args.n_iter, atk_vn=atk_vn, dfd_vn=dfd_vn, plot=True, ratio=args.ratio,
                  use_pred=True))
        print("Time:", time.time() - st)
    else:
        ps = []
        avs = []
        dvs = []

        n_samples = 20
        for i in range(n_samples):
            # p = np.random.random()
            p = i / n_samples
            ps.append(p)
        ps.append(1.0)
        # ps.append(0.5)

        st = time.time()
        with Pool(12) as p:
            vs = list(p.map(Runner(args, payoff=env.payoff, atk_vn=atk_vn, dfd_vn=dfd_vn), ps))
            avs, dvs, ex, exp = map(list, zip(*vs))
            # for i, p in enumerate(ps):
            #     print("\t".join(["{}"] * 7).format(p, ex[i][0][0], ex[i][0][1], ex[i][1], exp[i][0][0], exp[i][0][1],
            #                                        exp[i][1]))

        print("Time:", time.time() - st)
        print(avs)
        print(dvs)

        df = pd.DataFrame(dict(p=ps, v=avs))
        sns.scatterplot(x="p", y="v", data=df)
        if args.save_plot:
            plt.savefig("plot/" + get_filename(args) + ".png", dpi=100)
        else:
            plt.show()

        if args.save_data:
            np.array([ps, avs, dvs]).dump("data/" + get_filename(args) + ".obj")


if __name__ == "__main__":
    main()
