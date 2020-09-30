from modules.pack import CubicSplinePack, Interp1dPack, Interp1dPackFast, Interp1dPackUltraFast, NNPack
from utils.utils import ts
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def _plot(data, atk_vn):
    def access(x):
        return atk_vn(ts([x]), ts(1e-2)).numpy()[0]

    ax = np.arange(0., 1., 0.01)
    ay = list(map(access, ax))

    df = pd.DataFrame(dict(belief=atk_vn.xs, value=np.array(atk_vn.ys)[:, 0]))
    sns.scatterplot(x="belief", y="value", data=df)
    plt.plot(ax, ay, color='r')
    plt.show()


def _load_vn(interp, data):
    interp_dict = {
        "linear": Interp1dPack,
        "linear_fast": Interp1dPackFast,
        "cubic": CubicSplinePack,
        "nn": NNPack
    }
    interp_pack = interp_dict[interp]
    vn = interp_pack(data)
    # atk_vn = interp_pack(data[[0, 1, 2], :].transpose())
    # dfd_vn = interp_pack(data[[0, 3], :].transpose())

    # _plot(data, atk_vn)

    return vn


def load_vn(filename, interp="linear_fast"):
    print("loading {}".format(filename))
    data = np.load(filename)
    if type(interp) == str:
        return _load_vn(interp, data)
    else:
        vns = []
        for _interp in interp:
            vn = _load_vn(_interp, data)
            vns.append(vn)
        return vns
