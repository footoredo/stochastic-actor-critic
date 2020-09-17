import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from utils.utils import pred


def run():
    n_iters = 100000

    atk_r = np.zeros((2, 2))

    def get_s(r):
        s = np.maximum(r, 0.)
        sr = s.sum()
        if sr < 1e-6:
            return np.array([0.5, 0.5])
        else:
            return s / sr

    def get_belief(s, a):
        b = np.array([s[0][a], s[1][a]])
        return b / b.sum()

    def get_atk_v(b, t):
        if b[t] < b[1 - t] - 1e-8:
            return 5
        elif b[t] > b[1 - t] + 1e-8:
            return -5
        else:
            return 0

    a_atk_s = np.zeros((2, 2))

    avs = []
    sum_t = 0
    for tt in range(n_iters):
        atk_s = []
        for i in range(2):
            atk_s.append(get_s(atk_r[i]))

        atk_v = np.zeros((2, 2))
        atk_av = np.zeros(2)
        for t in range(2):
            for a in range(2):
                belief = get_belief(atk_s, a)
                atk_v[t][a] = get_atk_v(belief, t) + (1 if a == t else 0)
                atk_av[t] += atk_v[t][a] * atk_s[t][a]

        for t in range(2):
            for a in range(2):
                atk_r[t][a] += atk_v[t][a] - atk_av[t]

        # w = max(tt - 250, 0)
        w = 1
        a_atk_s = a_atk_s * sum_t + np.array(atk_s) * w
        sum_t += w

        if sum_t > 0:
            a_atk_s /= sum_t

            av = 0.
            for t in range(2):
                for a in range(2):
                    a_belief = get_belief(a_atk_s, a)
                    # print(get_atk_v(a_belief, t))
                    av += (get_atk_v(a_belief, t) + (1 if a == t else 0)) * 0.5 * a_atk_s[t][a]
            # print(av, a_atk_s)
            avs.append(av)
        else:
            avs.append(0)

    def test(b):
        x1 = avs[n_iters // 2] - b
        x2 = avs[n_iters - 1] - b
        k = np.log(x2 / x1) / np.log((n_iters - 1) / (n_iters // 2))
        a = x2 / (n_iters - 1) ** k
        diff = 0.
        for i in range(n_iters // 2, n_iters):
            diff += a * i ** k + b - avs[i]
        return diff

    # x = np.arange(0, 1, 0.01)
    # y = list(map(test, x))
    # plt.plot(x, y)
    # plt.show()

    b = pred(n_iters // 2, n_iters, avs)

    df = pd.DataFrame(dict(
        iter=list(range(n_iters)) + list(range(n_iters)),
        value=list(0.5 - np.array(avs)) + [abs(b - 0.5)] * n_iters,
        name=["av"] * n_iters + ["pred"] * n_iters
    ))

    fig, ax = plt.subplots()
    ax.set(yscale="log", ylim=[0.0000001, 100])
    ax.set(xscale="log")
    sns.lineplot(x="iter", y="value",  hue="name", data=df, ax=ax)
    plt.show()


if __name__ == "__main__":
    run()
