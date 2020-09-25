import numpy as np
from env.tagging_game import TaggingGame
from multiprocessing import Process, Value, Pipe
import time
import joblib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from modules.tagging_cfr_reader import ApproximatedReader
from functools import partial
import seaborn as sns
import pandas as pd


def regret_to_strategy(regret):
    strategy = np.maximum(regret, 0.)
    s = strategy.sum()
    if s < 1e-6:
        strategy = np.ones_like(strategy) / strategy.shape[0]
    else:
        strategy /= s
    return strategy


def bayes(belief, atk_s, a):
    belief = belief * atk_s[:, a]
    if belief.sum() < 1e-5:
        ret = np.ones_like(belief) / belief.shape[0]
    else:
        ret = belief / belief.sum()

    return ret


def calc_rew(env, reader, belief, tp, opp_pos, pro_pos, a, b):
    new_opp_pos, new_pro_pos, opp_reward, pro_reward = env.step(tp, opp_pos, pro_pos, a, b)
    if reader is not None:
        if any(np.isnan(belief)):
            belief = np.ones_like(belief) / belief.shape[0]
        _, _, opp_v, pro_v, _, _ = reader.access(belief, new_opp_pos, new_pro_pos)
        opp_reward += opp_v[tp]
        pro_reward += pro_v

    return opp_reward, pro_reward


def regret_matching(env, belief, opp_pos, pro_pos, n_iter, reader, conn):
    # if next_v is not None:
    #     reader = ApproximatedReader(next_v)
    # else:
    #     reader = None
    _calc_rew = partial(calc_rew, env, reader)
    opp_regret = np.zeros((2, 4))
    pro_regret = np.zeros(5)
    opp_av = np.zeros((2, 4))
    pro_av = np.zeros(5)

    ws = 0
    for tt in range(n_iter):
        opp_strategy = np.zeros((2, 4))
        for i in range(2):
            opp_strategy[i] = regret_to_strategy(opp_regret[i])
        pro_strategy = regret_to_strategy(pro_regret)
        w = max(tt - n_iter // 4, 0)
        # w = 1
        if ws + w > 0:
            opp_av = (opp_av * ws + opp_strategy * w) / (ws + w)
            pro_av = (pro_av * ws + pro_strategy * w) / (ws + w)
        ws += w

        opp_cfv = np.zeros((2, 4))
        opp_v = np.zeros(2)
        for i in range(2):
            for a in range(4):
                for b in range(5):
                    nb = bayes(belief, opp_strategy, a)
                    opp_r, _ = _calc_rew(nb, i, opp_pos, pro_pos, a, b)
                    opp_cfv[i][a] += opp_r * pro_strategy[b]
                    opp_v[i] += opp_r * opp_strategy[i, a] * pro_strategy[b]

        for i in range(2):
            for a in range(4):
                opp_regret[i, a] += opp_cfv[i, a] - opp_v[i]

        for i in range(2):
            opp_strategy[i] = regret_to_strategy(opp_regret[i])

        pro_cfv = np.zeros(5)
        pro_v = 0.
        for b in range(5):
            for i in range(2):
                for a in range(4):
                    nb = bayes(belief, opp_strategy, a)
                    _, pro_r = _calc_rew(nb, i, opp_pos, pro_pos, a, b)
                    pro_cfv[b] += pro_r * belief[i] * opp_strategy[i, a]
                    pro_v += pro_r * belief[i] * opp_strategy[i, a] * pro_strategy[b]

        for b in range(5):
            pro_regret[b] += pro_cfv[b] - pro_v

    opp_v = np.zeros(2)
    opp_cfv = np.zeros((2, 4))
    pro_v = 0.
    pro_cfv = np.zeros(5)

    for i in range(2):
        for a in range(4):
            for b in range(5):
                nb = bayes(belief, opp_av, a)
                if b == 0:
                    print(i, a, nb)
                opp_r, pro_r = _calc_rew(nb, i, opp_pos, pro_pos, a, b)
                # print(i, a, b, opp_r, pro_r)
                opp_v[i] += opp_r * opp_av[i][a] * pro_av[b]
                opp_cfv[i, a] += opp_r * pro_av[b]
                pro_v += pro_r * opp_av[i][a] * pro_av[b] * belief[i]
                pro_cfv[b] += pro_r * opp_av[i][a] * belief[i]

    if conn is not None:
        conn.send((opp_av, pro_av, opp_v, pro_v, opp_cfv, pro_cfv))

    return opp_av, pro_av, opp_v, pro_v, opp_cfv, pro_cfv


class Runner(object):
    def __init__(self, env, n_iter, reader):
        self.env = env
        self.n_iter = n_iter
        self.reader = reader

    def __call__(self, belief, opp_pos, pro_pos, conn):
        return regret_matching(self.env, belief, opp_pos, pro_pos, self.n_iter, self.reader, conn)


def execute_parallel(processes, conns):
    for p in processes:
        p.start()

    results = []
    for conn in conns:
        results.append(conn.recv())

    for p in processes:
        p.join()

    return results


def select(objects, indices):
    return [objects[i] for i in indices]


def run_parallel(env, n_iter, reader, jobs):
    print("Total jobs:", len(jobs))

    runner = Runner(env, n_iter, reader)
    # processes = []
    # conns = []
    # for job in jobs:
    #     parent_conn, child_conn = Pipe()
    #     process = Process(target=runner, args=job + (child_conn,))
    #     processes.append(process)
    #     conns.append(parent_conn)

    n_par = 12
    chunks = np.array_split(range(len(jobs)), (len(jobs) + n_par - 1) // n_par)
    results = []
    conns = []
    for i in range(n_par):
        conns.append(Pipe())

    for chunk in chunks:
        print("executing", chunk)
        st = time.time()
        chunk_jobs = select(jobs, chunk)
        processes = []
        chunk_conns = []
        for i, job in enumerate(chunk_jobs):
            parent_conn, child_conn = conns[i] 
            process = Process(target=runner, args=job + (child_conn,))
            processes.append(process)
            chunk_conns.append(parent_conn)
        cur_results = execute_parallel(processes, chunk_conns)
        results += cur_results
        print("Time:", time.time() - st)

    return results


def run(env, n_samples, n_iter):
    n = env.size
    ps = np.array(range(n_samples)) / (n_samples - 1)

    jobs = []
    for p in ps:
        # print("Solving belief ({}, {})".format(p, 1 - p))
        for opp_x in range(n):
            for opp_y in range(n):
                # print("opponent position: ({}, {})".format(opp_x, opp_y))
                for pro_x in range(n):
                    for pro_y in range(n // 2):
                        # print("protagonist position: ({}, {})".format(pro_x, pro_y))
                        # print(regret_matching(env, np.array([p, 1 - p]), np.array([opp_x, opp_y]),
                        #                       np.array([pro_x, pro_y]), n_iter))
                        jobs.append((np.array([p, 1 - p]), np.array([opp_x + 0.5, opp_y + 0.5]),
                                     np.array([pro_x + 0.5, pro_y + 0.5])))

    results = run_parallel(env, n_iter, jobs)

    joblib.dump(results, "tagging-{}_cfr.obj".format(env.size))


def interactive(env, n_iter, reader):
    while True:
        try:
            inputs = input().split()
            b = float(inputs[0])
            ox, oy, px, py = map(float, inputs[1:])
            print(regret_matching(env, np.array([b, 1 - b]), np.array([ox, oy]), np.array([px, py]), n_iter, reader, None))
        except ValueError:
            print("value error!")


def plot(env, p, n_samples, n_iter):
    xs = []
    ys = []
    vs = []
    jobs = []
    for _ in range(n_samples):
        # opp_pos = np.random.randn(2) * env.size
        opp_pos = np.array([4., 8.])
        pro_pos = np.random.rand(2) * np.array([env.size, env.size / 2])
        # print(pro_pos)
        jobs.append((np.array([p, 1 - p]), opp_pos, pro_pos))
        xs.append(pro_pos[0])
        ys.append(pro_pos[1])

    results = run_parallel(env, n_iter, None, jobs)
    for _, _, _, v, _, _ in results:
        vs.append(v)

    fig, ax = plt.subplots()
    scat = ax.scatter(xs, ys, c=vs, s=50, marker='o')
    fig.colorbar(scat)

    plt.show()


def plot2(env, n_samples, n_iter):
    xs = []
    vs = []
    ns = []
    jobs = []
    # opp_pos = np.random.rand(2) * np.array([env.size, env.size])
    # pro_pos = np.random.rand(2) * np.array([env.size, env.size / 2])
    opp_pos = np.array([4.0, 4.5])
    pro_pos = np.array([4.0, 4.0])
    print("opp_pos:", opp_pos)
    print("pro_pos:", pro_pos)
    for i in range(n_samples):
        p = i / (n_samples - 1)
        # opp_pos = np.random.randn(2) * env.size
        # print(pro_pos)
        jobs.append((np.array([p, 1 - p]), opp_pos, pro_pos))
        xs.append(p)
        ns.append("atk-0")
        # xs.append(p)
        # ns.append("atk-1")
        # xs.append(p)
        # ns.append("dfd")

    results = run_parallel(env, n_iter, None, jobs)
    for _, _, atk_v, dfd_v, _, _ in results:
        vs.append(atk_v[0])
        # vs.append(atk_v[1])
        # vs.append(dfd_v)

    def approximate(xx):
        sv = 0.
        sw = 0.
        for i, x in enumerate(xs):
            w = 1. / (1e-2 + np.square(x - xx))
            sv += w * results[i][2][0]
            sw += w
        return sv / sw

    ax = np.arange(0., 1., 0.01)
    ay = list(map(approximate, ax))

    # df2 = pd.DataFrame(dict(belief=ax, value=ay))

    df = pd.DataFrame(dict(belief=xs, value=vs, name=ns))

    # fig, axs = plt.subplots(ncols=2)
    sns.scatterplot(x="belief", y="value", hue="name", data=df)
    # sns.lineplot(x="belief", y="value", data=df2, ax=axs[1])
    plt.plot(ax, ay, color='r')
    plt.show()


def train_net(env, n_samples, n_iter):
    jobs = []
    for _ in range(n_samples):
        p = np.random.rand(1)
        opp_pos = np.random.rand(2) * np.array([env.size, env.size])
        pro_pos = np.random.rand(2) * np.array([env.size, env.size / 2])
        jobs.append((np.array([p, 1 - p]), opp_pos, pro_pos))
   
    results = run_parallel(env, n_iter, jobs)
   
    joblib.dump(jobs, "tmp-jobs.obj")
    joblib.dump(results, "tmp-results.obj")
    # jobs = joblib.load("tmp-jobs.obj")
    # results = joblib.load("tmp-results.obj")

    dim = 32
    model = nn.Sequential(
        nn.Linear(5, dim),
        nn.ReLU(),
        nn.Linear(dim, 3)
    )

    loss_fn = nn.MSELoss()
    lr = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    xs = []
    ys = []

    for i in range(n_samples):
        belief, opp_pos, pro_pos = jobs[i]
        opp_av, pro_av, opp_v, pro_v, opp_cfv, pro_cfv = results[i]
        xs.append(np.concatenate((belief[0], opp_pos, pro_pos)))
        ys.append(np.concatenate((opp_v, pro_v)))

    xs = torch.tensor(np.array(xs), dtype=torch.float)
    ys = torch.tensor(np.array(ys), dtype=torch.float)

    batch_size = 50
    n_train = int(n_samples * 0.8)
    for t in range(n_train // batch_size):
        batch_x = xs[t * batch_size: (t + 1) * batch_size]
        batch_y = ys[t * batch_size: (t + 1) * batch_size]
        y_pred = model(batch_x)
        loss = loss_fn(y_pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 10 == 9:
            y_test_pred = model(xs[n_train:])
            test_loss = loss_fn(y_test_pred, ys[n_train:])
            print(t, loss.item(), test_loss.item())

    while True:
        inputs = input().split()
        b = float(inputs[0])
        ox, oy, px, py = map(float, inputs[1:])
        opp_av, pro_av, opp_v, pro_v = \
            regret_matching(env, np.array([b, 1 - b]), np.array([ox, oy]), np.array([px, py]), n_iter, None)
        x = torch.tensor([np.concatenate(([b], [ox, oy], [px, py]))], dtype=torch.float)
        print("Truth:", opp_v, pro_v)
        print("Predicted:", model(x)[0])


def main():
    np.set_printoptions(precision=3, suppress=True)
    env = TaggingGame(8)
    # reader = ApproximatedReader("tagging-8_cfr")
    reader = None
    # run(env, 7, 200)
    interactive(env, 1000, reader)
    # plot(env, 0.1, 2000, 100)
    # plot2(env, 11, 100)
    # train_net(env, 10000, 100)


if __name__ == "__main__":
    main()