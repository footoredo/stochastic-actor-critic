import numpy as np


def one_hot(n, i):
    x = np.zeros(n)
    if n > i >= 0:
        x[i] = 1.
    return x


def update_belief(belief, prob):
    a = belief * prob
    if np.sum(a) < 1e-8:
        a = np.ones_like(a)
    return a / np.sum(a)


class SecurityGame(object):
    def __init__(self, n_slots, n_types, prior, n_rounds, value_low=5., value_high=10., seed=None):
        self.n_slots = n_slots
        self.n_types = n_types
        self.prior = prior if prior is not None else np.random.rand(n_types)
        self.prior /= np.sum(self.prior)
        self.n_rounds = n_rounds
        self.seed = seed

        if seed is not None:
            np.random.seed(int(seed))

        value_range = value_high - value_low
        self.atk_rew = np.random.rand(n_types, n_slots) * value_range + value_low
        self.atk_pen = -np.random.rand(n_types, n_slots) * value_range - value_low
        self.dfd_rew = np.random.rand(n_slots) * value_range + value_low
        self.dfd_pen = -np.random.rand(n_slots) * value_range - value_low

        self.payoff = np.zeros((n_types, n_slots, n_slots, 2), dtype=np.float32)
        for t in range(n_types):
            for i in range(n_slots):
                for j in range(n_slots):
                    if i == j:
                        self.payoff[t, i, j, 0] = self.atk_pen[t, i]
                        self.payoff[t, i, j, 1] = self.dfd_rew[j]
                    else:
                        self.payoff[t, i, j, 0] = self.atk_rew[t, i]
                        self.payoff[t, i, j, 1] = self.dfd_pen[j]

        self.ob_len = [n_types * 2 + n_rounds, n_types + n_rounds]

        self.belief = None
        self.i_round = None
        self.atk_type = None

    def _get_pub_ob(self, belief, i_round):
        return np.concatenate([belief, one_hot(self.n_rounds, i_round)])

    def _get_atk_ob(self, atk_type, belief, i_round):
        return np.concatenate([self._get_pub_ob(belief, i_round), one_hot(self.n_types, atk_type)])

    def _get_dfd_ob(self, belief, i_round):
        return self._get_pub_ob(belief, i_round)

    def _get_ob(self, atk_type, belief, i_round):
        return [self._get_atk_ob(atk_type, belief, i_round), self._get_dfd_ob(belief, i_round)]

    def generate_belief(self):
        x = [0.] + sorted(np.random.rand(self.n_types - 1).tolist()) + [1.]
        for i in range(self.n_types):
            self.prior[i] = x[i + 1] - x[i]

    def reset(self):
        self.generate_belief()
        self.belief = np.copy(self.prior)
        self.i_round = 0
        self.atk_type = np.random.choice(range(self.n_types), p=self.prior)

        return self._get_ob(self.atk_type, self.belief, self.i_round)

    def step(self, actions):
        atk_act, dfd_act = actions
        rews = self.payoff[self.atk_type, atk_act, dfd_act]
        self.i_round += 1
        done = self.i_round >= self.n_rounds
        # self.belief = update_belief()
        return self._get_ob(self.atk_type, self.belief, self.i_round), rews, done, None
