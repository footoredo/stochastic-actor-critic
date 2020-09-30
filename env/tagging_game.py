import numpy as np


def calc_dis(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


class TaggingGame(object):
    def __init__(self, size=8, next_v=None):
        self.size = size
        self.tag_lim = 1.5 if size == 4 else 2.5
        self.ally_base = np.array([0.5, size - 0.5])
        self.enemy_base = np.array([size - 0.5, size - 0.5])

    def _x_inbound(self, x):
        return 0 <= x <= self.size

    def _p_inbound(self, p, is_pro):
        if not (self._x_inbound(p[0]) and self._x_inbound(p[1])):
            return False
        if is_pro and p[1] > self.size / 2.:
            return False
        return True

    def _move(self, p, d, is_pro):
        move_d = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
        new_p = p + move_d[d]
        if self._p_inbound(new_p, is_pro):
            return new_p
        else:
            return p

    def step(self, opp_type, opp_pos, pro_pos, opp_action, pro_action, prob_count=1):
        opp_reward = 0.
        pro_reward = 0.

        if pro_action == 4:  # tag
            pro_reward -= 0.2
            if calc_dis(pro_pos, opp_pos) < self.tag_lim:  # tag success
                opp_reward -= 10.
                if opp_type == 0:  # ally
                    pro_reward -= 20.
                else:  # enemy
                    pro_reward += 10.

        if pro_action == 5:  # prob
            pro_reward -= 0.25 * prob_count

        new_opp_pos = self._move(opp_pos, opp_action, False)
        if pro_action < 4:
            new_pro_pos = self._move(pro_pos, pro_action, True)
        else:
            new_pro_pos = pro_pos

        opp_reward -= 0.25 * np.power(calc_dis(new_opp_pos, self.ally_base if opp_type == 0 else self.enemy_base), 0.4)
        pro_reward -= 0.25 * np.power(calc_dis(new_pro_pos, new_opp_pos), 0.4)

        return new_opp_pos, new_pro_pos, opp_reward, pro_reward
