import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def main():
    one = np.load("one-net.data")
    two = np.load("two-nets.data")
    n = len(one)

    df = pd.DataFrame(dict(iteration=list(range(n)) + list(range(n)),
                           value=list(np.abs(one[:, 0, 0] - one[:, 1, 0])) + list(np.abs(two[:, 0, 0] - two[:, 1, 0])),
                           name=["one"] * n + ["two"] * n))
    fig, ax = plt.subplots()
    ax.set(xscale="log")
    ax.set(yscale="log", ylim=[0.0001, 100])
    sns.lineplot(x="iteration", y="value", hue="name", data=df, ax=ax)
    plt.show()


if __name__ == "__main__":
    main()
