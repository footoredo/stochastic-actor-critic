import numpy as np
import torch
import torch.nn as nn


def ts(v):
    return torch.tensor(v, dtype=torch.float)


class ValueNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, is_opp):
        super(ValueNet, self).__init__()

        if is_opp:
            self.linear1 = nn.ModuleList([nn.Linear(input_dim - 1, hidden_dim), nn.Linear(input_dim - 1, hidden_dim)])
            self.linear2 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)])
            self.linear3 = nn.ModuleList([nn.Linear(hidden_dim, output_dim), nn.Linear(hidden_dim, output_dim)])
        else:
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
            self.linear3 = nn.Linear(hidden_dim, output_dim)

        self.is_opp = is_opp

    def forward(self, _input):
        if not self.is_opp:
            x = self.linear1(_input)
            x = self.linear2(torch.tanh(x))
            x = self.linear3(torch.tanh(x))
        else:
            w1 = _input[:, 0:1]
            w0 = 1. - w1
            _input = _input[:, 1:]
            x = w0 * self.linear1[0](_input) + w1 * self.linear1[1](_input)
            x = w0 * self.linear2[0](x) + w1 * self.linear2[1](x)
            x = w0 * self.linear3[0](x) + w1 * self.linear3[1](x)
        return x

    def reinitialize(self):
        if self.is_opp:
            for i in range(2):
                self.linear1[i].reset_parameters()
                self.linear2[i].reset_parameters()
                self.linear3[i].reset_parameters()
        else:
            self.linear1.reset_parameters()
            self.linear2.reset_parameters()
            self.linear3.reset_parameters()

    def zero_(self):
        with torch.no_grad():
            if self.is_opp:
                for i in range(2):
                    self.linear1[i].weight.zero_()
                    self.linear1[i].bias.zero_()
                    self.linear2[i].weight.zero_()
                    self.linear2[i].bias.zero_()
                    self.linear3[i].weight.zero_()
                    self.linear3[i].bias.zero_()
            else:
                self.linear1.weight.zero_()
                self.linear1.bias.zero_()
                self.linear2.weight.zero_()
                self.linear2.bias.zero_()
                self.linear3.weight.zero_()
                self.linear3.bias.zero_()


def _extend_input(_input, is_opp):
    if not isinstance(_input, np.ndarray):
        _input = np.array(_input)
    if (is_opp and _input.shape[0] == 8) or (not is_opp and _input.shape[0] == 6):
        return _input
    if is_opp:
        tp = _input[0]
        _input = _input[1:]
    else:
        tp = None
    p = _input[0]
    opp_pos = _input[1:3]
    pro_pos = _input[3:]

    tag_available = np.sum(np.square(pro_pos - opp_pos)) < 2.5 * 2.5
    if opp_pos[1] > 4.:
        tag_available = False

    new_inputs = [[p], opp_pos, pro_pos, opp_pos - pro_pos,
                  [1. if tag_available else 0.]]
    if is_opp:
        new_inputs = [[1., 0.] if tp < 0.5 else [0., 1.]] + new_inputs

    return np.concatenate(new_inputs)


def get_strategy(adv):
    if np.sum(np.abs(adv)) < 1e-5:
        adv = np.ones_like(adv)
    if np.max(adv) <= 0.:
        a = np.argmax(adv)
        adv = np.zeros_like(adv)
        adv[a] = 1.
    s = np.maximum(adv, 0.)
    return s / s.sum()


hidden_dim = 32
pro_input_n = 8
opp_input_n = 10


class CFRProActor(object):
    def __init__(self):
        self.nets = [None]
        pro_adv_net = ValueNet(pro_input_n, hidden_dim, 6, False)
        for n_round in range(1, 6):
            avg_pro_adv_net = ValueNet(pro_input_n, hidden_dim, 6, False)

            n = 40
            w = 1. / n

            avg_pro_adv_net.zero_()
            for i in range(n):
                pro_adv_net.load_state_dict(torch.load("models/{}/None/pro_adv_{}.net".format(n_round, i)))

                for pa, pb in zip(avg_pro_adv_net.parameters(), pro_adv_net.parameters()):
                    pa.data += pb.data * w

            self.nets.append(avg_pro_adv_net)

    def strategy(self, n_round, belief, opp_x, opp_y, pro_x, pro_y):
        pro_input = ts([_extend_input([belief[0], opp_y + 0.5, opp_x + 0.5, pro_y + 0.5, pro_x + 0.5], False)])
        adv = self.nets[n_round](pro_input)[0].detach().numpy()
        strategy = get_strategy(adv)[:5]
        strategy /= strategy.sum()
        return strategy

    def act(self, n_round, belief, opp_x, opp_y, pro_x, pro_y):
        strategy = self.strategy(n_round, belief, opp_x, opp_y, pro_x, pro_y)
        return np.random.choice(range(5), p=strategy)


class CFROppActor(object):
    def __init__(self):
        self.nets = [None]
        opp_adv_net = ValueNet(opp_input_n, hidden_dim, 4, True)
        for n_round in range(1, 6):
            avg_opp_adv_net = ValueNet(opp_input_n, hidden_dim, 4, True)

            n = 40
            w = 1. / n
            # w = 1. / 5

            avg_opp_adv_net.zero_()
            for i in range(n):
                opp_adv_net.load_state_dict(torch.load("models/{}/None/opp_adv_{}.net".format(n_round, i)))

                for pa, pb in zip(avg_opp_adv_net.parameters(), opp_adv_net.parameters()):
                    pa.data += pb.data * w

            self.nets.append(avg_opp_adv_net)

    def strategy(self, n_round, tp, belief, opp_x, opp_y, pro_x, pro_y):
        opp_input = ts([_extend_input([tp, belief[0], opp_y + 0.5, opp_x + 0.5, pro_y + 0.5, pro_x + 0.5], True)])
        adv = self.nets[n_round](opp_input)[0].detach().numpy()
        strategy = get_strategy(adv)[:4]
        strategy /= strategy.sum()
        return strategy

    def act(self, n_round, tp, belief, opp_x, opp_y, pro_x, pro_y):
        strategy = self.strategy(n_round, tp, belief, opp_x, opp_y, pro_x, pro_y)
        return np.random.choice(range(4), p=strategy)


if __name__ == "__main__":
    pro_actor = CFRProActor()
    opp_actor = CFROppActor()

    while True:
        r, p, ox, oy, px, py = map(float, input().split())
        r = int(r + 0.5)
        print(pro_actor.strategy(r, [p], ox, oy, px, py))
        print(opp_actor.strategy(r, 0, [p], ox, oy, px, py))
        print(opp_actor.strategy(r, 1, [p], ox, oy, px, py))
