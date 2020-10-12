import numpy as np
import matplotlib.pyplot as plt
from utils.utils import ts
from utils.tools import load_vn
import pandas as pd
import seaborn as sns


def plot_true():
    # vn = load_vn("data/demo-1.atk_vn.obj", interp="nn")
    # vx = np.arange(0, 1. + 0.01, 0.01)
    # vy = list(map(lambda x: vn(ts([x]), 1e-2)[0].item(), vx))

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    x1 = np.arange(0, 1 / 3, 0.01)
    y1 = np.ones_like(x1) * 5.
    x2 = np.arange(1 / 3, 2 / 3, 0.01)
    y2 = np.zeros_like(x2)
    x3 = np.arange(2 / 3, 1. + 0.01, 0.01)
    y3 = np.ones_like(x3) * 5.

    sx1 = [x1[-1], x2[0]]
    sy1 = [5.0, 0.0]
    sx2 = [x2[-1], x3[0]]
    sy2 = [0.0, 5.0]

    ax.set_ylim(-0.2, 5.2)
    # axes[1].set_ylim(-0.1, 5.1)
    # axes[2].set_ylim(-0.1, 5.1)

    ax.plot(x1, y1, c="purple")
    ax.plot(x2, y2, c="purple")
    ax.plot(x3, y3, c="purple")
    ax.plot(sx1, sy1, c="purple", dashes=(5, 5))
    ax.plot(sx2, sy2, c="purple", dashes=(5, 5))

    # axes[1].plot(vx, vy, c="purple")

    plt.savefig("exposing-true.pdf")


def plot_approx():
    vn = load_vn("data/demo-1.atk_vn.obj", interp="nn")
    vx = np.arange(0, 1. + 0.01, 0.01)
    vy = list(map(lambda x: vn(ts([x]), 1e-2)[0].item(), vx))

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))

    ax.set_ylim(-0.2, 5.2)

    ax.plot(vx, vy, c="purple")

    plt.savefig("exposing-approx.pdf")


def plot_linear():
    vn = load_vn("data/demo-1.atk_vn.obj", interp="linear_fast")
    vx = np.arange(0, 1. + 0.01, 0.01)
    vy = list(map(lambda x: vn(ts([x]), 1e-2)[0].item(), vx))

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))

    ax.set_ylim(-0.2, 5.2)

    ax.plot(vx, vy, c="purple")

    plt.savefig("exposing-linear.pdf")


def plot_curve(name):
    data = np.load("exposing-{}.atk_ass.data".format(name))
    # print(data.shape)
    n = data.shape[0]
    x = list(range(n))
    y1 = data[:, 0, 0]
    y2 = data[:, 1, 0]

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.plot(x, y1, color="purple")
    ax.plot(x, y2, color="orange")

    # df = pd.DataFrame(dict(
    #     iteration=x + x,
    #     value=list(y1) + list(y2),
    #     type=["type 1"] * n + ["type 2"] * n
    # ))
    #
    # sns.lineplot(x="iteration", y="value", hue="type", data=df)

    plt.show()


def plot_curve_all():
    data = np.load("exposing-{}.atk_ass.data".format("cfr"))
    # print(data.shape)
    n = data.shape[0]
    x = list(range(n))
    y_cfr_1 = data[1001:, 0, 0]
    y_cfr_2 = data[1001:, 1, 0]
    # print(y_cfr_1[-1])
    data = np.load("exposing-{}.atk_ass.data".format("pg"))
    y_pg_1 = data[:, 0, 0]
    y_pg_2 = data[:, 1, 0]
    data = np.load("exposing-{}.atk_ass.data".format("pg-dt"))
    y_pd_1 = data[:, 0, 0]
    y_pd_2 = data[:, 1, 0]

    fig, axes = plt.subplots(1, 3, figsize=(5, 2))

    for i in range(3):
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].set_xlim(-200, n + 200)

    axes[0].set_ylabel("prob. of action 1")

    axes[0].set_xlabel("TIPG")
    axes[1].set_xlabel("TICFR\n#iterations")
    axes[2].set_xlabel("TIPG$^-$")

    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])

    axes[1].plot(x[1001:], y_cfr_1, color="purple")
    axes[1].plot(x[1001:], y_cfr_2, color="orange")
    axes[0].plot(x, y_pg_1, color="purple")
    axes[0].plot(x, y_pg_2, color="orange")
    axes[2].plot(x, y_pd_1, color="purple")
    axes[2].plot(x, y_pd_2, color="orange")

    axes[2].legend(("type 1", "type 2"))

    plt.tight_layout(pad=0.2)

    plt.savefig("exposing-curve.pdf")


if __name__ == "__main__":
    plot_curve_all()
