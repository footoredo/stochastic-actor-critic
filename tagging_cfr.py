import numpy as np
from env.tagging_game import TaggingGame
from multiprocessing import Process, Value, Pipe
import time
import joblib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.tagging_cfr_reader import ApproximatedReader
from functools import partial
import seaborn as sns
import pandas as pd
from utils.utils import ts, dt


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
    if belief.sum() < 1e-3:
        ret = np.ones_like(belief) / belief.shape[0]
    else:
        ret = belief / belief.sum()

    return ret


def calc_rew(env, reader, belief, tp, opp_pos, pro_pos, a, b):
    new_opp_pos, new_pro_pos, opp_reward, pro_reward = env.step(tp, opp_pos, pro_pos, a, b)
    if reader is not None:
        if any(np.isnan(belief)):
            belief = np.ones_like(belief) / belief.shape[0]
        if b == 5:
            return -10000, -10000
            for r in range(2):
                if r == 0:
                    belief[tp] *= 0.8
                    belief[1 - tp] *= 0.2
                else:
                    belief[tp] *= 0.2
                    belief[1 - tp] *= 0.8
                belief /= belief.sum()
                _, _, opp_v, pro_v, _, _ = reader.access(belief, new_opp_pos, new_pro_pos)
                p = 0.8 if r == 0 else 0.2
                opp_reward += opp_v[tp] * p
                pro_reward += pro_v * p
        else:
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
    pro_regret = np.zeros(6)
    opp_av = np.zeros((2, 4))
    pro_av = np.zeros(6)

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
                for b in range(6):
                    nb = bayes(belief, opp_strategy, a)
                    opp_r, _ = _calc_rew(nb, i, opp_pos, pro_pos, a, b)
                    opp_cfv[i][a] += opp_r * pro_strategy[b]
                    opp_v[i] += opp_r * opp_strategy[i, a] * pro_strategy[b]

        for i in range(2):
            for a in range(4):
                opp_regret[i, a] += opp_cfv[i, a] - opp_v[i]

        for i in range(2):
            opp_strategy[i] = regret_to_strategy(opp_regret[i])

        pro_cfv = np.zeros(6)
        pro_v = 0.
        for b in range(6):
            for i in range(2):
                for a in range(4):
                    nb = bayes(belief, opp_strategy, a)
                    _, pro_r = _calc_rew(nb, i, opp_pos, pro_pos, a, b)
                    pro_cfv[b] += pro_r * belief[i] * opp_strategy[i, a]
                    pro_v += pro_r * belief[i] * opp_strategy[i, a] * pro_strategy[b]

        for b in range(6):
            pro_regret[b] += pro_cfv[b] - pro_v

    opp_v = np.zeros(2)
    opp_cfv = np.zeros((2, 4))
    pro_v = 0.
    pro_cfv = np.zeros(6)

    for i in range(2):
        for a in range(4):
            for b in range(6):
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


def run(env, n_samples, n_iter, reader=None):
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

    results = run_parallel(env, n_iter, reader, jobs)

    joblib.dump(results, "tagging-{}_cfr_2.obj".format(env.size))


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


class ValueNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ValueNet, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, _input):
        x = self.linear1(_input)
        x = self.linear2(torch.tanh(x))
        x = self.linear3(torch.tanh(x))
        return x

    def reinitialize(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.linear3.reset_parameters()

    def zero_(self):
        with torch.no_grad():
            self.linear1.weight.zero_()
            self.linear1.bias.zero_()
            self.linear2.weight.zero_()
            self.linear2.bias.zero_()
            self.linear3.weight.zero_()
            self.linear3.bias.zero_()


def act_on_adv(adv, rng=None):
    if rng is None:
        rng = np.random
    # print(adv)
    if np.sum(np.abs(adv)) < 1e-5:
        return rng.choice(range(adv.shape[0]))
    if np.max(adv) <= 0.:
        return np.argmax(adv)
    else:
        s = np.maximum(adv, 0.)
        return rng.choice(range(s.shape[0]), p=s / s.sum())


def train_v_net(n_batches, batch_size, lr, v_net: ValueNet, data: pd.DataFrame, verbose=False, reinit=True, n_passes=3,
                device=None):
    if reinit:
        v_net.reinitialize()
    loss_fn = nn.MSELoss()
    v_optimizer = torch.optim.Adam(v_net.parameters(), lr=lr)

    test_data = data[n_batches * batch_size:]
    test_input = ts(np.stack(test_data["input"].to_list(), axis=0)).to(device)
    test_output = ts(test_data["reward"].to_numpy()).to(device)

    for T in range(n_passes):
        # print("    pass #{}".format(T))
        for i in range(n_batches):
            _input = np.stack(data["input"][i * batch_size: (i + 1) * batch_size].to_list(), axis=0)
            _output = data["reward"][i * batch_size: (i + 1) * batch_size].to_numpy(dtype=np.float)
            __input = ts(_input).to(device)
            _pred = v_net(__input)

            v_loss = loss_fn(_pred.squeeze().to(device), ts(_output).to(device))

            if verbose:
                if (i + 1) % 20 == 0:
                    test_pred = v_net(test_input)
                    print(loss_fn(test_pred.squeeze(), test_output).item())

            v_optimizer.zero_grad()
            v_loss.backward()
            v_optimizer.step()


def gen_adv_data(v_data: pd.DataFrame, v_net: ValueNet):
    v_pred = dt(v_net(ts(v_data["input"])))
    adv_data = v_data.copy()
    for i in range(len(adv_data["input"])):
        delta = v_pred[i, 0]
        adv_data["reward"][i] -= delta
    return adv_data


def train_adv_net(n_batches, batch_size, lr, adv_net: ValueNet, data: pd.DataFrame, verbose=False, reinit=True,
                  n_passes=3, device=None):
    if reinit:
        adv_net.reinitialize()
    loss_fn = nn.MSELoss()
    adv_optimizer = torch.optim.Adam(adv_net.parameters(), lr=lr)

    test_data = data[n_batches * batch_size:]
    test_input = ts(np.stack(test_data["input"].to_list(), axis=0)).to(device)
    test_action = torch.tensor(test_data["action"].to_numpy(dtype=np.int), dtype=torch.long).unsqueeze(-1).to(device)
    test_output = ts(test_data["reward"].to_numpy()).to(device)

    for T in range(n_passes):
        # print("    pass #{}".format(T))
        for i in range(n_batches):
            # batch_data = data.sample(n=batch_size)
            _input = np.stack(data["input"][i * batch_size: (i + 1) * batch_size].to_list(), axis=0)
            _action = data["action"][i * batch_size: (i + 1) * batch_size].to_numpy(dtype=np.int)
            _output = data["reward"][i * batch_size: (i + 1) * batch_size].to_numpy(dtype=np.float)
            # _input = np.stack(batch_data["input"].to_list(), axis=0)
            # _action = batch_data["action"].to_numpy(dtype=np.int)
            # _output = batch_data["reward"].to_numpy(dtype=np.float)
            _pred = adv_net(ts(_input).to(device))

            adv_loss = loss_fn(torch.gather(_pred, 1,
                                            torch.tensor(_action, dtype=torch.long).unsqueeze(-1)).squeeze().to(device),
                               ts(_output).to(device))

            if verbose:
                if (i + 1) % 20 == 0:
                    test_pred = adv_net(test_input)
                    test_loss = loss_fn(torch.gather(test_pred, 1, test_action).squeeze(), test_output)
                    print(test_loss.item())

            adv_optimizer.zero_grad()
            adv_loss.backward()
            adv_optimizer.step()


def model_executor(net, batch_size, total_jobs, main_conn, worker_conns, job_queue):
    remain_jobs = total_jobs
    while remain_jobs > 0:
        n_jobs = min(batch_size, remain_jobs)
        ids = []
        inputs = []
        for i in range(n_jobs):
            _id, _input = job_queue.get(block=True)
            ids.append(_id)
            inputs.append(_input)

        results = net(ts(inputs)).detach().numpy()
        for i in range(n_jobs):
            worker_conns[ids[i]].send(results[i])
        remain_jobs -= n_jobs
        # print(remain_jobs)


# def _sample1(_id, size, n, main_conn, pro_queue, opp_queue):
#     for i in range(n):
#         p = [0.2]
#         opp_pos = np.random.randint(size, size=2) + 0.5
#         pro_pos = np.random.randint(size, size=2) % np.array([size, size // 2]) + 0.5
#         # opp_pos = np.random.rand(2) * np.array([env.size, env.size])
#         # pro_pos = np.random.rand(2) * np.array([env.size, env.size / 2])
#         opp_type = np.random.choice([0, 1], p=[p[0], 1 - p[0]])
#
#         pro_input = np.concatenate((p, opp_pos, pro_pos))
#         opp_input = np.concatenate(([opp_type], p, opp_pos, pro_pos))
#
#         extended_pro_input = _extend_input(pro_input, False)
#         extended_opp_input = _extend_input(opp_input, True)
#
#         pro_queue.put((_id, extended_pro_input))
#         opp_queue.put((_id, extended_opp_input))
#
#         main_conn.send((opp_pos, pro_pos, pro_input, opp_input, opp_type))
#
#
# def _sample2(_id, env, n, main_conn, pro_conn, opp_conn):
#     for i in range(n):
#         pro_action = act_on_adv(pro_conn.recv())
#         opp_action = act_on_adv(opp_conn.recv())
#
#         opp_pos, pro_pos, pro_input, opp_input, opp_type = main_conn.recv()
#
#         _, _, opp_r, pro_r = env.step(opp_type, opp_pos, pro_pos, opp_action, pro_action)
#
#         main_conn.send((pro_input, pro_action, pro_r, opp_input, opp_action, opp_r))


def _sample(_id, env, n, main_conn, pro_queue, pro_conn, opp_queue, opp_conn):
    rng = np.random.RandomState()
    for i in range(n):
        # p = [0.3]
        p = np.random.rand(1)
        opp_pos = rng.randint(env.size, size=2) + 0.5
        pro_pos = rng.randint(env.size, size=2) % np.array([env.size, env.size // 2]) + 0.5
        # opp_pos = np.random.rand(2) * np.array([env.size, env.size])
        # pro_pos = np.random.rand(2) * np.array([env.size, env.size / 2])
        opp_type = rng.choice([0, 1], p=[p[0], 1 - p[0]])

        pro_input = np.concatenate((p, opp_pos, pro_pos))
        opp_input = np.concatenate(([opp_type], p, opp_pos, pro_pos))

        extended_pro_input = _extend_input(pro_input, False)
        extended_opp_input = _extend_input(opp_input, True)

        pro_queue.put((_id, extended_pro_input))
        opp_queue.put((_id, extended_opp_input))

        pro_action = act_on_adv(pro_conn.recv(), rng)
        opp_action = act_on_adv(opp_conn.recv(), rng)

        # print(_id)

        _, _, opp_r, pro_r = env.step(opp_type, opp_pos, pro_pos, opp_action, pro_action)

        # print(pro_input, pro_action, pro_r, opp_input, opp_action, opp_r)

        main_conn.send((pro_input, pro_action, pro_r, opp_input, opp_action, opp_r))


def sample(env, n_samples, pro_adv_net, opp_adv_net, pro_v_buffer, opp_v_buffer):
    if n_samples == 0:
        return
    batch_size = 50
    concurrency = 50
    n = n_samples // concurrency
    assert n_samples % n == 0

    import torch.multiprocessing as mp

    sst = time.time()

    pro_adv_net.share_memory()
    opp_adv_net.share_memory()

    buffer = mp.Queue()
    sema = mp.Semaphore(10)

    opp_worker_recv_conns = []
    opp_worker_send_conns = []
    pro_worker_recv_conns = []
    pro_worker_send_conns = []
    main_worker_conns = []
    worker_main_conns = []
    worker_locks = []
    for i in range(concurrency):
        conn1, conn2 = mp.Pipe()
        opp_worker_recv_conns.append(conn1)
        opp_worker_send_conns.append(conn2)
        conn1, conn2 = mp.Pipe()
        pro_worker_recv_conns.append(conn1)
        pro_worker_send_conns.append(conn2)
        conn1, conn2 = mp.Pipe()
        main_worker_conns.append(conn1)
        worker_main_conns.append(conn2)
        worker_locks.append(mp.Lock())

    main_pro, pro_main = mp.Pipe()
    main_opp, opp_main = mp.Pipe()

    pro_queue = mp.Queue()
    pro_p = mp.Process(target=model_executor, args=(pro_adv_net, batch_size, n_samples,
                                                    pro_main, pro_worker_send_conns, pro_queue))
    opp_queue = mp.Queue()
    opp_p = mp.Process(target=model_executor, args=(opp_adv_net, batch_size, n_samples,
                                                    opp_main, opp_worker_send_conns, opp_queue))

    pro_p.start()
    opp_p.start()

    ps = []
    for i in range(concurrency):
        p = mp.Process(target=_sample, args=(i, env, n, worker_main_conns[i], pro_queue,
                                             pro_worker_recv_conns[i], opp_queue, opp_worker_recv_conns[i]))
        p.start()
        ps.append(p)

    st = time.time()
    for j in range(n):
        for i in range(concurrency):
            pro_input, pro_action, pro_r, opp_input, opp_action, opp_r = main_worker_conns[i].recv()
            pro_v_buffer["input"].append(pro_input)
            pro_v_buffer["action"].append(pro_action)
            pro_v_buffer["reward"].append(pro_r)
            opp_v_buffer["input"].append(opp_input)
            opp_v_buffer["action"].append(opp_action)
            opp_v_buffer["reward"].append(opp_r)
        # if (j + 1) % 100 == 0:
        #     dti = time.time() - st
        #     print(dti, (n - j) * dti / 100)
        #     st = time.time()

    for p in ps:
        p.join()

    pro_p.join()
    opp_p.join()

    # cur_i = 0
    # while cur_i < n_samples:
    #     batch_rem = min(batch_size, n_samples - cur_i)
    #     ps1 = []
    #     ps2 = []
    #     for i in range(batch_rem // n):
    #         # worker_locks[i].acquire()
    #         p1 = mp.Process(target=_sample1, args=(i, env.size, n, worker_main_conns[i], pro_queue, opp_queue))
    #         ps1.append(p1)
    #         p2 = mp.Process(target=_sample2, args=(i, env, n, worker_main_conns[i], pro_worker_recv_conns[i],
    #                                                opp_worker_recv_conns[i]))
    #         ps2.append(p2)
    #     st_i = 0
    #     im_data = []
    #     while st_i * n < batch_rem:
    #         sst = time.time()
    #         n_now = min(concurrency, batch_rem // n - st_i)
    #         for i in range(n_now):
    #             ps1[st_i + i].start()
    #         for i in range(n_now):
    #             d = []
    #             for j in range(n):
    #                 d.append(main_worker_conns[st_i + i].recv())
    #             im_data.append(d)
    #             ps1[st_i + i].join()
    #         st_i += n_now
    #         print(1, st_i, time.time() - sst)
    #
    #     main_pro.recv()
    #     main_opp.recv()
    #
    #     st_i = 0
    #     while st_i * n < batch_rem:
    #         sst = time.time()
    #         n_now = min(concurrency, batch_rem // n - st_i)
    #         for i in range(n_now):
    #             ps2[st_i + i].start()
    #             for j in range(n):
    #                 main_worker_conns[st_i + i].send(im_data[st_i + i][j])
    #         for i in range(n_now):
    #             for j in range(n):
    #                 pro_input, pro_action, pro_r, opp_input, opp_action, opp_r = main_worker_conns[st_i + i].recv()
    #
    #                 pro_v_buffer["input"].append(pro_input)
    #                 pro_v_buffer["action"].append(pro_action)
    #                 pro_v_buffer["reward"].append(pro_r)
    #                 opp_v_buffer["input"].append(opp_input)
    #                 opp_v_buffer["action"].append(opp_action)
    #                 opp_v_buffer["reward"].append(opp_r)
    #
    #             ps2[st_i + i].join()
    #
    #         st_i += n_now
    #         print(2, st_i, time.time() - sst)
    #         # print(st_i)
    #
    #     cur_i += batch_rem

    # for i in range(n_samples):
    #     pro_input, pro_action, pro_r, opp_input, opp_action, opp_r = buffer.get(block=True)
    #     # print(i)
    #
    #     pro_v_buffer["input"].append(pro_input)
    #     pro_v_buffer["action"].append(pro_action)
    #     pro_v_buffer["reward"].append(pro_r)
    #     opp_v_buffer["input"].append(opp_input)
    #     opp_v_buffer["action"].append(opp_action)
    #     opp_v_buffer["reward"].append(opp_r)
    #
    # for p in ps:
    #     p.join()

    print("Sample time:", time.time() - sst)


def _extend_input(_input, is_opp):
    if not isinstance(_input, np.ndarray):
        _input = np.array(_input)
    if (is_opp and _input.shape[0] == 8) or (not is_opp and _input.shape[0] == 6):
        return _input
    if is_opp:
        tp = _input[0]
        _input = _input[1:]
    else:
        tp = None
    p = _input[0]
    opp_pos = _input[1:3]
    pro_pos = _input[3:]

    tag_available = np.sum(np.square(pro_pos - opp_pos)) < 2.5 * 2.5
    if opp_pos[1] > 4.:
        tag_available = False

    new_inputs = [[p], opp_pos, pro_pos, opp_pos - pro_pos,
                  [1. if tag_available else 0.]]
    if is_opp:
        new_inputs = [[1., 0.] if tp < 0.5 else [0., 1.]] + new_inputs

    return np.concatenate(new_inputs)


def extend_input(buffer, is_opp):
    for i in range(len(buffer["input"])):
        buffer["input"][i] = _extend_input(buffer["input"][i], is_opp)
        if not is_opp:
            x = buffer["input"][i]
            dis = x[5]
            # if buffer["action"][i] == 4:
            #     print(x, buffer["reward"][i])

        # print(buffer["input"][i])


def save_buffer(buffer: dict, filename):
    import joblib
    import copy
    _buffer = copy.deepcopy(buffer)
    for k in _buffer.keys():
        _buffer[k] = np.array(_buffer[k])
    joblib.dump(_buffer, filename)


def load_buffer(filename):
    import joblib
    buffer = joblib.load(filename)
    for k in buffer.keys():
        if isinstance(buffer[k], np.ndarray):# and len(buffer[k].shape) > 1:
            buffer[k] = list(buffer[k])
        # print(type(buffer[k]), type(buffer[k][0]))
    return buffer


def _buffer_add(buffer, entry, size, tot):
    if len(buffer["input"]) < size:
        buffer["input"].append(entry[0])
        buffer["action"].append(entry[1])
        buffer["reward"].append(entry[2])
    else:
        replace = np.random.randint(tot)
        if replace < size:
            buffer["input"][replace] = entry[0]
            buffer["action"][replace] = entry[1]
            buffer["reward"][replace] = entry[2]


def buffer_add(buffer, other_buffer, size, tot):
    # print(other_buffer["input"])
    for i in range(len(other_buffer["input"])):
        _buffer_add(buffer, (other_buffer["input"][i], other_buffer["action"][i], other_buffer["reward"][i]), size, tot)


def cfr(env):
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")
    device = cpu
    # print(device)
    hidden_dim = 32
    pro_input_n = 8
    opp_input_n = 10
    pro_v_net = ValueNet(pro_input_n, hidden_dim, 1).to(device)
    pro_adv_net = ValueNet(pro_input_n, hidden_dim, 6).to(device)
    opp_v_net = ValueNet(opp_input_n, hidden_dim, 1).to(device)
    opp_adv_net = ValueNet(opp_input_n, hidden_dim, 4).to(device)

    # print(pro_v_net(ts(np.random.randn(6)).to(device)))

    pro_adv_net.zero_()
    opp_adv_net.zero_()

    n_iter = 1
    batch_size = 5000
    n_batches = 20
    lr = 5e-2
    verbose = False
    n_passes = 50
    test_points = [(0.3, 3.5, 3.5, 3.5, 2.5), (0.3, 3.5, 3.5, 3.5, 0.5), (0.4, 3.5, 3.5, 3.5, 2.5)]
    load = True
    save = True

    pro_adv_df = pd.DataFrame(dict(input=[], action=[], reward=[]))
    opp_adv_df = pd.DataFrame(dict(input=[], action=[], reward=[]))

    def display(p, ox, oy, px, py):
        pro_input = ts(_extend_input([p, ox, oy, px, py], False)).to(device)
        opp_input_0 = ts(_extend_input([0., p, ox, oy, px, py], True)).to(device)
        opp_input_1 = ts(_extend_input([1., p, ox, oy, px, py], True)).to(device)
        print("pro_v:", pro_v_net(pro_input)[0].item())
        print("opp_v:", opp_v_net(opp_input_0)[0].item(),
              opp_v_net(opp_input_1)[0].item())
        print("pro_adv:", pro_adv_net(pro_input).detach().numpy())
        print("opp_adv:", opp_adv_net(opp_input_0).detach().numpy(),
              opp_adv_net(opp_input_1).detach().numpy())

    for p in test_points:
        display(*p)

    for it in range(n_iter):
        print("Iteration #", it)

        if load:
            pro_v_buffer = load_buffer("models/pro_v_buffer_{}.obj".format(it))
            # print(type(pro_v_buffer["input"]))
            opp_v_buffer = load_buffer("models/opp_v_buffer_{}.obj".format(it))
        else:
            pro_v_buffer = dict(input=[], action=[], reward=[])
            opp_v_buffer = dict(input=[], action=[], reward=[])

        pro_adv_buffer = dict(input=[], action=[], reward=[])
        opp_adv_buffer = dict(input=[], action=[], reward=[])

        n_train = batch_size * n_batches
        n_test = int(n_train * 0.1)
        n_samples = n_train + n_test

        sample(env, n_samples - len(pro_v_buffer["input"]), pro_adv_net, opp_adv_net, pro_v_buffer, opp_v_buffer)

        # for i in range(100):
        #     print(pro_v_buffer["input"][i], pro_v_buffer["action"][i], pro_v_buffer["reward"][i])

        if save:
            save_buffer(pro_v_buffer, "models/pro_v_buffer_{}.obj".format(it))
            save_buffer(opp_v_buffer, "models/opp_v_buffer_{}.obj".format(it))

        extend_input(pro_v_buffer, False)
        extend_input(opp_v_buffer, True)

        pro_v_net.reinitialize()
        opp_v_net.reinitialize()

        pro_v_df = pd.DataFrame(pro_v_buffer)
        opp_v_df = pd.DataFrame(opp_v_buffer)

        pro_v_df.sample(frac=1).reset_index(drop=True)  # shuffle
        opp_v_df.sample(frac=1).reset_index(drop=True)  # shuffle

        st = time.time()

        _train_v_net = partial(train_v_net, n_batches, batch_size, lr)
        print("Training pro_v_net")
        _train_v_net(pro_v_net, pro_v_df, verbose, True, n_passes)
        print("Training opp_v_net")
        _train_v_net(opp_v_net, opp_v_df, verbose, True, n_passes)

        _train_adv_net = partial(train_adv_net, n_batches, batch_size, lr)
        buffer_add(pro_adv_buffer, gen_adv_data(pro_v_buffer, pro_v_net), n_samples, (it + 1) * n_samples)
        pro_adv_df = pd.DataFrame(pro_adv_buffer)
        # print(pro_adv_df.shape)
        # pro_adv_df.sample(frac=1).reset_index(drop=True)  # shuffle
        print("Training pro_adv_net")
        _train_adv_net(pro_adv_net, pro_adv_df, verbose, True, n_passes)

        # while True:
        #     display(0.1, 3.5, 3.5, 3.5, 3.5)
        #     prompt = input()
        #     if prompt == "cont":
        #         break
        #     _train_adv_net(pro_adv_net, pro_adv_df, verbose, False)

        buffer_add(opp_adv_buffer, gen_adv_data(opp_v_buffer, opp_v_net), n_samples, (it + 1) * n_samples)
        opp_adv_df = pd.DataFrame(opp_adv_buffer)
        # opp_adv_df = opp_adv_df.append(gen_adv_data(opp_v_df, opp_v_net), ignore_index=True)
        # opp_adv_df.sample(frac=1).reset_index(drop=True)  # shuffle
        print("Training opp_adv_net")
        _train_adv_net(opp_adv_net, opp_adv_df, verbose, True, n_passes)

        if save:
            save_buffer(pro_adv_buffer, "models/pro_adv_buffer_{}.obj".format(it))
            save_buffer(opp_adv_buffer, "models/opp_adv_buffer_{}.obj".format(it))


        print("Time: {}s".format(time.time() - st))

        # if save:
        #     pro_adv_df.to_pickle("models/pro_adv_df_{}.obj".format(it))
        #     opp_adv_df.to_pickle("models/opp_adv_df_{}.obj".format(it))

        for p in test_points:
            display(*p)

        torch.save(pro_v_net.state_dict(), "models/pro_v_{}.net".format(it))
        torch.save(pro_adv_net.state_dict(), "models/pro_adv_{}.net".format(it))
        torch.save(opp_v_net.state_dict(), "models/opp_v_{}.net".format(it))
        torch.save(opp_adv_net.state_dict(), "models/opp_adv_{}.net".format(it))

    while True:
        x = input()
        if x == "cont":
            break
        try:
            p, ox, oy, px, py = map(float, x.split())
            display(p, ox, oy, px, py)
        except ValueError:
            continue


def main():
    np.set_printoptions(precision=3, suppress=True)
    env = TaggingGame(8)
    # reader = ApproximatedReader("tagging-8_cfr")
    # reader = None
    cfr(env)
    # run(env, 11, 1000, reader)
    # interactive(env, 1000, reader)
    # plot(env, 0.1, 2000, 100)
    # plot2(env, 11, 100)
    # train_net(env, 10000, 100)


def test_time():
    n = 10000
    net = ValueNet(6, 32, 4)

    data = np.random.randn(n, 6)
    inputs = list(map(ts, list(data)))

    st = time.time()
    s1 = np.zeros(4)
    for i in range(n):
        s1 += net(inputs[i]).detach().numpy()
    print(time.time() - st)

    inputs = ts(data)

    st = time.time()
    s2 = np.zeros(4)
    ret = net(inputs).detach().numpy()
    for i in range(n):
        s2 += ret[i]

    print(time.time() - st)
    print(s2 - s1)


def _numpy_test(lock):
    lock.acquire()
    print("1232")
    return 0


def test_numpy_parallel():
    n = 1000

    import torch.multiprocessing as mp

    st = time.time()

    ps = []
    lock = mp.Lock()
    lock.acquire()
    for i in range(n):
        for _ in range(500):
            p = mp.Process(target=_numpy_test, args=(lock,))
            p.start()
            # p.join()
            ps.append(p)

        st = time.time()
        b = np.random.randn(120)
        for _ in range(1000):
            a = np.random.randn(120)
            b += a
        print(time.time() - st, b.sum())

    for i in range(n):
        ps[i].join()

    print(time.time() - st)


if __name__ == "__main__":
    main()
