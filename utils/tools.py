from modules.pack import CubicSplinePack, Interp1dPack, Interp1dPackFast, Interp1dPackUltraFast
import numpy as np


def load_vn(filename, interp="linear_fast"):
    interp_dict = {
        "linear": Interp1dPack,
        "linear_fast": Interp1dPackFast,
        "cubic": CubicSplinePack
    }
    interp_pack = interp_dict[interp]
    data = np.load(filename)
    atk_vn = interp_pack(data[[0, 1, 2], :].transpose())
    dfd_vn = interp_pack(data[[0, 3], :].transpose())
    return atk_vn, dfd_vn
