import matplotlib.pyplot as plt
import numpy as np
from functools import partial


def f(eps, x):
    dis = x - 0.1
    if dis <= 0:
        w = 1. / eps * (1. - dis / (eps / 1e-2))
    else:
        w = 1. / (eps + dis * dis)
    return w


def plot1():
    x = np.arange(0., 1., 0.01)
    y = list(map(partial(f, 1e-2), x))
    plt.plot(x, y)
    y = list(map(partial(f, 1e-5), x))
    # plt.plot(x, y, color="r")
    plt.show()


def plot2():
    n_iter = 1000
    eps_k = np.log(1000.) / n_iter
    start_eps = 1e-2
    xs = list(range(n_iter))
    ys = []
    for x in xs:
        ys.append(start_eps / np.exp(eps_k * x))
    fig, ax = plt.subplots()
    # ax.set(yscale="log")
    ax.plot(xs, ys)
    plt.show()


if __name__ == "__main__":
    plot2()
