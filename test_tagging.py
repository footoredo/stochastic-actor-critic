import torch
from utils.utils import *
from tagging_cfr import ValueNet, _extend_input


def get_strategy(adv):
    if np.sum(np.abs(adv)) < 1e-5:
        adv = np.ones_like(adv)
    if np.max(adv) <= 0.:
        a = np.argmax(adv)
        adv = np.zeros_like(adv)
        adv[a] = 1.
    s = np.maximum(adv, 0.)
    return s / s.sum()


def test_net(it):
    hidden_dim = 32
    pro_input_n = 8
    opp_input_n = 10

    pro_v_net = ValueNet(pro_input_n, hidden_dim, 1)
    pro_adv_net = ValueNet(pro_input_n, hidden_dim, 6)
    opp_v_net = ValueNet(opp_input_n, hidden_dim, 1)
    opp_adv_net = ValueNet(opp_input_n, hidden_dim, 4)

    # pro_v_net.load_state_dict(torch.load("models/pro_v_{}.net".format(it)))
    # opp_v_net.load_state_dict(torch.load("models/opp_v_{}.net".format(it)))
    # pro_adv_net.load_state_dict(torch.load("models/pro_adv_{}.net".format(it)))
    # opp_adv_net.load_state_dict(torch.load("models/opp_adv_{}.net".format(it)))

    def display(p, ox, oy, px, py):
        pro_input = ts(_extend_input([p, ox, oy, px, py], False))
        opp_input_0 = ts(_extend_input([0., p, ox, oy, px, py], True))
        opp_input_1 = ts(_extend_input([1., p, ox, oy, px, py], True))
        # print("pro_v:", pro_v_net(pro_input)[0].item())
        # print("opp_v:", opp_v_net(opp_input_0)[0].item(),
        #       opp_v_net(opp_input_1)[0].item())
        # print("pro_adv:", pro_adv_net(pro_input).detach().numpy())
        # print("pro_strat:", get_strategy(pro_adv_net(pro_input).detach().numpy()))
        # print("opp_adv:", opp_adv_net(opp_input_0).detach().numpy(),
        #       opp_adv_net(opp_input_1).detach().numpy())
        # print("opp_strat:", get_strategy(opp_adv_net(opp_input_0).detach().numpy()),
        #       get_strategy(opp_adv_net(opp_input_1).detach().numpy()))
        pro_as = np.zeros(6)
        opp_as = np.zeros((2, 4))
        n = 100
        w = 1. / (n // 2)
        for i in range(n // 2, n):
            pro_adv_net.load_state_dict(torch.load("models/pro_adv_{}.net".format(i)))
            opp_adv_net.load_state_dict(torch.load("models/opp_adv_{}.net".format(i)))

            pro_s = get_strategy(pro_adv_net(pro_input).detach().numpy())
            pro_as += pro_s * w
            opp_s_0 = get_strategy(opp_adv_net(opp_input_0).detach().numpy())
            opp_s_1 = get_strategy(opp_adv_net(opp_input_1).detach().numpy())
            opp_as += np.array([opp_s_0, opp_s_1]) * w

        print("pro:", pro_as)
        print("opp:", opp_as)

    print("load done.")
    while True:
        x = input()
        try:
            p, ox, oy, px, py = map(float, x.split())
            display(p, ox, oy, px, py)
        except ValueError:
            continue


if __name__ == "__main__":
    test_net(99)
