import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import linregress


def detach_ts(x):
    if type(x) == torch.Tensor:
        return x.detach()
    else:
        return ts(x)


def get_parser(name):
    parser = argparse.ArgumentParser(description=name)
    parser.add_argument('--env-seed', type=int, default=5410, metavar='N',
                        help='random seed (default: 5410)')
    parser.add_argument('--seed', type=int, metavar='N',
                        help='random seed ')
    parser.add_argument('--prior', type=float, nargs='+')
    parser.add_argument('--n-slots', type=int, default=2)
    parser.add_argument('--n-types', type=int, default=2)
    parser.add_argument('--n-iter', type=int, default=10000)
    parser.add_argument('--n-rounds', type=int, default=1)
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--zero-sum', action="store_true", default=False)
    parser.add_argument('--all', action="store_true", default=False)
    parser.add_argument('--save-data', action="store_true", default=False)
    parser.add_argument('--save-plot', action="store_true", default=False)
    parser.add_argument('--demo', action="store_true", default=False)
    return parser


def to_np(v):
    if type(v) == torch.Tensor:
        return dt(v)
    else:
        return np.array(v)


def ts(v):
    return torch.tensor(v, dtype=torch.float)


def dt(v):
    return v.detach().numpy()


def regularize(a):
    if check_none(a):
        return None
    a = a.clip(0., 1.)
    return a / np.sum(a)


def get_filename(args, n_rounds=None, n_iter=None, n_samples=None):
    r = n_rounds if n_rounds is not None else args.n_rounds
    i = n_iter if n_iter is not None else args.n_iter
    n = n_samples if n_samples is not None else args.n_samples
    if args.demo:
        name = "demo"
    # elif args.sep:
    #     name = "sec-sep"
    else:
        name = "sec"
    if args.cfr:
        name += "-cfr"
    if args.demo:
        filename = "{}-{}".format(name, r)
    else:
        # filename = "{}-{}-{}-{}-{}-{}-{}-{}".format(name, args.env_seed, args.n_types, args.n_slots, args.n_types, r, i, n)
        ilename = "{}-{}-{}-{}-{}-{}-{}".format(name, args.env_seed, args.n_types, args.n_slots, args.n_types, r, i)
    return filename


def analyse_wave(data):
    # if 1. - np.mean(data) < 1e-2 or np.mean(data) - 0. < 1e-2:
    #     return None
    eps = 1e-4
    n = len(data)
    turns = []
    last_turn = 0
    last_turn_data = 0.
    distances = []
    heights = []
    direction = 0
    increase_count = 0
    expand_points = []
    expand_degree = []
    v_max = data[0]
    v_min = data[0]
    for i in range(1, n):
        v_max = max(v_max, data[i])
        v_min = min(v_min, data[i])
        if data[i] - data[i - 1] < -eps:
            curd = -1
        elif data[i] - data[i - 1] > eps:
            curd = 1
        else:
            curd = direction
        if curd != direction:
            # if v_max is None and curd == -1:
            #     v_max = data[i]
            # if v_min is None and curd == 1:
            #     v_min = data[i]
            # if v_max is not None and data[i] > v_max + 1e-5:
            #     expand_degree.append(data[i] - v_max)
            #     increase_count += 1
            #     v_max = data[i]
            #     expand_points.append(i)
            # if v_min is not None and data[i] < v_min - 1e-5:
            #     # expand_degree.append(v_min - data[i])
            #     increase_count += 1
            #     v_min = data[i]
            #     # expand_points.append(i)
            direction = curd
            turns.append(i)
            distances.append(i - last_turn)
            heights.append(np.abs(data[i] - last_turn_data))
            # print(i, distances[-1], heights[-1])
            last_turn = i
            last_turn_data = data[i]

    n = len(turns)
    print(heights)
    # print(expand_points)
    # print(expand_degree)
    # print(np.mean(expand_degree[n // 2:]))

    # fig, ax = plt.subplots()
    # df = pd.DataFrame(dict(turn=turns[2:],
    #                        size=heights[2:]))
    # ax.set(yscale="log")
    # sns.scatterplot(x="turn", y="size", data=df, ax=ax)
    # plt.show()

    # print(n, np.mean(heights[n // 2:]))
    if n < 4 and (v_max is None or v_min is None or v_max - v_min < 0.5):
        return None

    if n > 2:
        slope, intercept, r_value, p_value, std_err = linregress(turns[2:], heights[2:])
    else:
        slope = 0.
    # print(slope)
    # if n > 2:
    #     heights = heights[2:]
    # print(np.max(heights) - np.min(heights))

    # if n >= 4:
    #     heights = heights[2:]
    #     n -= 2
    # print("num waves:", n)
    # print("height diff:", np.mean(heights[n // 2:]) - np.mean(heights[:n//2]))
    # print("avg height:", np.mean(heights))
    # hd = np.mean(heights[n // 2:]) - np.mean(heights[:n//2]) if n >= 2 else 0.
    ha = np.mean(heights[2:]) if n > 2 else np.mean(heights)
    # print(heights)
    return n, slope, v_max - v_min


def _fit_error(x, y):
    # slope, intercept, r_value, p_value, std_err = linregress(x, y)
    # print(slope, intercept, r_value, p_value, std_err)
    # return std_err

    n = len(x)
    n1 = n // 3
    n2 = n - n // 3

    x1 = np.mean(x[:n1])
    y1 = np.mean(y[:n1])
    x2 = np.mean(x[n1:n2])
    y2 = np.mean(y[n1:n2])
    x3 = np.mean(x[n2:])
    y3 = np.mean(y[n2:])

    return np.abs(np.cross([x2 - x1, y2 - y1], [x3 - x2, y3 - y2]))


def _pred_2(n, n_iters, avs):
    data = np.array(avs[-n:])
    its = np.log(list(range(n_iters - n, n_iters)))
    xs = np.arange(0.1045, 0.1046, 0.00001)
    ys = [_fit_error(its, np.log(data - x)) for x in xs]
    plt.plot(xs, ys)
    plt.show()


def _pred(n, n_iters, avs):
    items = [[], [], [], []]
    # for i in range(n // 2):
    #     items[2 * (i % 2)].append(n_iters - i - 1)
    # for i in range(n // 2, n):
    #     items[2 * (i % 2) + 1].append(n_iters - i - 1)

    n = (int(n) // 4) * 4

    if np.std(avs[-n:]) < 1e-6:
        return np.mean(avs[-n:])

    dis = n // 4
    for i in range(dis):
        items[0].append(n_iters - 1 - i)
    for i in range(dis, dis * 2):
        items[1].append(n_iters - 1 - i)
    for i in range(dis * 2, dis * 3):
        items[2].append(n_iters - 1 - i)
    for i in range(dis * 3, dis * 4):
        items[3].append(n_iters - 1 - i)

    up = 0.
    down = 0.
    for i in items[0]:
        up += avs[i]
    for i in items[1]:
        up -= avs[i]
    for i in items[2]:
        down += avs[i]
    for i in items[3]:
        down -= avs[i]

    v = up / down
    # print(up, down)

    def _testk(k):
        _up = 0.
        _down = 0.
        for i in items[0]:
            _up += i ** k
        for i in items[1]:
            _up -= i ** k
        for i in items[2]:
            _down += i ** k
        for i in items[3]:
            _down -= i ** k
        return _up, _down

    def testk(k):
        u, d = _testk(k)
        return u / d - v

    # x = np.arange(-10.0, -0.001, 0.01)
    # y = list(map(testk, x))
    # plt.plot(x, y)
    # plt.show()

    l = -10.0
    r = -0.001
    for _ in range(20):
        m = (l + r) / 2
        if testk(m) > 0:
            r = m
        else:
            l = m

    if l > -0.001 - 1e-5 or l < -10 + 1e-5:
        a = 0.0
        l = 0.0
        b = np.mean(avs[-n:])
    else:
        # print(l, testk(l), _testk(l))
        u, d = _testk(l)
        a = np.mean([up / u, down / d])
        # print(up / u, down / d)
        bs = []
        for i in range(n_iters - n, n_iters):
            bs.append(avs[i] - a * i ** l)
        b = np.mean(bs)
        # a = np.mean([(x1 - x2) / (i1 ** l - i2 ** l), (x3 - x4) / (i3 ** l - i4 ** l)])
        # b = np.mean([x1 - a * i1 ** l, x2 - a * i2 ** l, x3 - a * i3 ** l, x4 - a * i4 ** l])

        # print(b, np.std(bs), avs[-1] - a * (n_iters - 1) ** l)
        # print(a, l, b)

    error = np.mean(np.abs(np.array(avs[-n:]) - np.array([a * (i ** l) + b for i in range(n_iters - n, n_iters)]))) / np.mean(np.abs(np.array(avs[-n:])))
    # print("a:", a, "l:", l, "error:", error, "b:", b)

    xs = np.array(list(range(n_iters - n, n_iters)))
    # ys = a * np.power(xs, l) + b
    # # print(len(list(xs) + list(xs)), len(list(ys) + list(avs[-n:])), len(["predict"] * n + ["truth"] * n), n)
    #
    # df = pd.DataFrame(dict(x=list(xs) + list(xs),
    #                        y=list(ys) + list(avs[-n:]),
    #                        name=["predict"] * n + ["truth"] * n))
    # sns.lineplot(x="x", y="y", hue="name", data=df)
    # plt.show()

    # if error > 1e-2:
    #     return None
    predict = a * np.power(100000, l) + b
    # print("predict", predict)
    return predict
    # return b


def check_none(a):
    if a is None:
        return True
    for x in list(a.reshape(-1)):
        if x is None:
            return True
    return False


def pred(n, n_iters, avs):
    try:
        return _pred(n, n_iters, avs)
    except ArithmeticError:
        return None
