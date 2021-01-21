import numpy as np


class DemoGame(object):
    def __init__(self, n_round):
        if n_round == 1:
            # atk_payoff = np.array([[[-2, 2], [-2, 2]], [[2, -2], [2, -2]]])
            # dfd_payoff = -atk_payoff
            atk_payoff = np.array([[[2, -1000], [0, 1]], [[2, -1000], [0, 1]]])
            dfd_payoff = np.array([[[2, 0], [-1000, 1]], [[2, 0], [-1000, 1]]])
        else:
            # atk_payoff = np.array([[[1, 1], [0, 0]], [[0, 0], [1, 1]]])
            atk_payoff = np.array([[[1, 1], [0, 0]], [[1, 1], [0, 0]]])
            # atk_payoff = np.array([[[2, -1], [-2, 1]], [[1, -2], [-1, 2]]])
            # dfd_payoff = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
            dfd_payoff = -atk_payoff

        self.payoff = np.array([atk_payoff, dfd_payoff]).transpose([1, 2, 3, 0])
