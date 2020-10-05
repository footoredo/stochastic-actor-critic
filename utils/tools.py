from modules.pack import CubicSplinePack, Interp1dPack, Interp1dPackFast, Interp1dPackUltraFast, NNPack, \
    DiscretePackFast
from utils.utils import ts
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def _plot(data, atk_vn):
    p = 0

    def access(x):
        return atk_vn(ts([x]), ts(1e-3)).numpy()[p]

    ax = np.arange(0., 1.01, 0.01)
    ay = list(map(access, ax))

    df = pd.DataFrame(dict(belief=atk_vn.xs, value=np.array(atk_vn.ys)[:, p]))
    sns.scatterplot(x="belief", y="value", data=df)
    plt.plot(ax, ay, color='r')
    plt.show()


def _load_vn(interp, data, verbose):
    interp_dict = {
        "linear": Interp1dPack,
        "linear_fast": Interp1dPackFast,
        "cubic": CubicSplinePack,
        "discrete": DiscretePackFast,
        "nn": NNPack
    }
    interp_pack = interp_dict[interp]
    # data.sort(axis=1)
    vn = interp_pack(data[data[:, 0].argsort()])
    # atk_vn = interp_pack(data[[0, 1, 2], :].transpose())
    # dfd_vn = interp_pack(data[[0, 3], :].transpose())

    if verbose:
        print(data)
        _plot(data, vn)

    return vn


def load_vn(filename, interp="linear_fast", verbose=False):
    print("loading {}".format(filename))
    data = np.load(filename)
    if type(interp) == str:
        return _load_vn(interp, data, verbose)
    else:
        vns = []
        for _interp in interp:
            vn = _load_vn(_interp, data, verbose)
            vns.append(vn)
        return vns
