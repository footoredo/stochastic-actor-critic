import numpy as np
from copy import deepcopy


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
    def __init__(self, n_slots, n_types, prior, n_rounds, beta=1.0, value_low=5., value_high=10., seed=None,
                 random_prior=False):
        self.rng = np.random.RandomState(seed=seed)

        self.n_slots = n_slots
        self.n_types = n_types
        self.prior = prior if prior is not None else np.zeros(n_types)
        self.prior /= np.sum(self.prior)
        self.n_rounds = n_rounds
        self.seed = seed
        self.random_prior = random_prior
        self.beta = beta

        value_range = value_high - value_low
        atk_rew = self.rng.rand(n_types, n_slots) * value_range + value_low
        atk_pen = -self.rng.rand(n_types, n_slots) * value_range - value_low
        dfd_rew = self.rng.rand(n_slots) * value_range + value_low
        dfd_pen = -self.rng.rand(n_slots) * value_range - value_low

        self.payoff = np.zeros((n_types, n_slots, n_slots, 2), dtype=np.float32)
        for t in range(n_types):
            for i in range(n_slots):
                for j in range(n_slots):
                    if i == j:
                        self.payoff[t, i, j, 0] = atk_pen[t, i]
                        # self.payoff[t, i, j, 1] = -atk_pen[t, i]
                        self.payoff[t, i, j, 1] = dfd_rew[j]
                    else:
                        self.payoff[t, i, j, 0] = atk_rew[t, i]
                        # self.payoff[t, i, j, 1] = -atk_rew[t, i]
                        self.payoff[t, i, j, 1] = dfd_pen[j]

        # print(self.payoff)
        self.ob_len = [n_types * 2 + n_rounds, n_types + n_rounds]

        self.belief = None
        self.i_round = None
        self.atk_type = None
        self.atk_vn = None
        self.dfd_vn = None
        self.attacker_strategy_exploiter = self._AttackerStrategyExploiter(self)
        self.defender_strategy_exploiter = self._DefenderStrategyExploiter(self)
        self.attacker_utility_calculator = self._AttackerUtilityCalculator(self)
        self.defender_utility_calculator = self._DefenderUtilityCalculator(self)

    def _get_pub_ob(self, belief, i_round):
        return np.concatenate([belief, one_hot(self.n_rounds, i_round)])

    def _get_atk_ob(self, atk_type, belief, i_round):
        return np.concatenate([self._get_pub_ob(belief, i_round), one_hot(self.n_types, atk_type)])

    def _get_dfd_ob(self, belief, i_round):
        return self._get_pub_ob(belief, i_round)

    def _get_ob(self, atk_type, belief, i_round):
        return [self._get_atk_ob(atk_type, belief, i_round), self._get_dfd_ob(belief, i_round)]

    def generate_belief(self):
        x = [0.] + sorted(self.rng.rand(self.n_types - 1).tolist()) + [1.]
        for i in range(self.n_types):
            self.prior[i] = x[i + 1] - x[i]

    def reset(self):
        if self.random_prior:
            self.generate_belief()
        self.belief = np.copy(self.prior)
        self.i_round = 0
        self.atk_type = self.rng.choice(range(self.n_types), p=self.prior)

        return self._get_ob(self.atk_type, self.belief, self.i_round)

    def step(self, actions):
        atk_act, dfd_act = actions
        rews = self.payoff[self.atk_type, atk_act, dfd_act]
        self.i_round += 1
        done = self.i_round >= self.n_rounds
        # self.belief = update_belief()
        return self._get_ob(self.atk_type, self.belief, self.i_round), rews, done, None

    def encode_history(self, history):
        return ','.join(map(str, history))

    def decode_history(self, encoded_history):
        if encoded_history == '':
            return []
        return list(map(int, encoded_history.split(',')))

    def _convert_attacker_strategy(self, attacker_strategy, defender_strategy):
        profile = [[] for _ in range(self.n_types)]

        def convert(t, history):
            s = dict()
            if len(history) < self.n_rounds:
                tmp = attacker_strategy(t, history)
                s[self.encode_history(history)] = tmp
                profile[t].append((str(history), tmp[0]))
                for a in range(self.n_slots):
                    s.update(convert(t, history + [a]))
            return s

        strategy = []
        for at in range(self.n_types):
            strategy.append(convert(at, []))
            profile[at] = sorted(profile[at], key=lambda x: (len(x[0]), x[0]))
            p = []
            for entry in profile[at]:
                p.append(entry[1])
            # print(at, p)
        return strategy

    def _convert_defender_strategy(self, attacker_strategy, defender_strategy):
        profile = []

        def convert(history):
            s = dict()
            if len(history) < self.n_rounds:
                # print(belief, history)
                tmp = defender_strategy(history)
                s[self.encode_history(history)] = tmp
                profile.append((str(history), tmp[0]))
                for a in range(self.n_slots):
                    s.update(convert(history + [a]))
            return s

        strategy = dict()
        strategy.update(convert([]))

        profile = sorted(profile, key=lambda x: (len(x[0]), x[0]))
        p = []
        for entry in profile:
            p.append(entry[1])
        # print(p)
        return strategy

    def calc_vn(self, attacker_strategy, defender_strategy, batch_size, train_steps):
        atk_vn = [ValueNetwork("newatk%d" % i, self.n_types, 64, 2) for i in range(self.n_types)]
        dfd_vn = ValueNetwork("newdfd", self.n_types, 64, 2)

        for i in range(train_steps):
            inputs = np.zeros(shape=(batch_size, self.n_types))
            atk_values = [np.zeros(shape=batch_size) for _ in range(self.n_types)]
            dfd_values = np.zeros(shape=batch_size)
            for j in range(batch_size):
                self.generate_belief()
                inputs[j] = prior = np.copy(self.prior)
                atk_action = [attacker_strategy.act(self._get_atk_ob(t, prior, 0)) for t in range(self.n_types)]
                dfd_action = defender_strategy.act(self._get_dfd_ob(prior, 0))
                atk_value = [self.get_atk_payoff(t, atk_action[t], dfd_action) for t in range(self.n_types)]
                atk_true_action = self.rng.choice(atk_action, 1, p=prior)
                dfd_value = self.get_def_payoff(atk_true_action, dfd_action, None)
                if self.atk_vn is not None and self.dfd_vn is not None:
                    for t in range(self.n_types):
                        atk_value[t] += self.atk_vn[t].calc(self.prior)
                    dfd_value += self.dfd_vn.calc(self.prior)
                    self.rounds_so_far = self.n_rounds - 1
                for t in range(self.n_types):
                    atk_values[t][j] = atk_value[t]
                dfd_values[j] = dfd_value
            for t in range(self.n_types):
                atk_vn[t].loss_step(inputs, atk_values[t])
            dfd_vn.loss_step(inputs, dfd_values)

            if (i + 1) % (train_steps // 10) == 0:
                for p in range(11):
                    prior = np.array([p / 10, 1 - p / 10])
                    print("Prior:", prior)
                    print("  Attacker:", [atk_vn[t].calc(prior) for t in range(self.n_types)])
                    print("  Defender:", dfd_vn.calc(prior))
                print("-----")

        return atk_vn, dfd_vn

    def _assess_strategies(self, strategies, verbose):
        if verbose:
            print("prior:", self.prior)
        attacker_strategy, defender_strategy = strategies
        tas = self._convert_attacker_strategy(attacker_strategy, defender_strategy)
        tds = self._convert_defender_strategy(attacker_strategy, defender_strategy)

        # print("vpred:", [attacker_strategy.vpred(self._get_atk_ob(i, self.prior, 0)) for i in range(self.n_types)],
        #       defender_strategy.vpred(self._get_dfd_ob(self.prior, 0)))

        def display(ts, t='?'):
            for k, v in ts.items():
                if len(k) < 3:
                    print(t, k, v)

        if verbose:
            print("Attacker:")
            for t in range(self.n_types):
                display(tas[t], str(t))
            print("Defender:")
            display(tds)

        def cut(s):
            if len(s) == 2:
                return [s[0]]
            else:
                return s

        for_sheet = sum([cut(tas[t][''].tolist()) for t in range(self.n_types)], []) + cut(tds[''].tolist())
        # return None

        atk_br = self.attacker_strategy_exploiter.run(tas, self.prior)
        def_br = self.defender_strategy_exploiter.run(tds)
        atk_u = self.attacker_utility_calculator.run(tas, tds, self.prior)
        def_u = self.defender_utility_calculator.run(tas, tds, self.prior)

        # print(def_br)
        # print(atk_u)
        # print(atk_br)
        # print(def_u)

        atk_result = []

        atk_pbne_eps = [0.] * self.n_types
        for t in range(self.n_types):
            for h, v in atk_u[t].items():
                atk_pbne_eps[t] = max(atk_pbne_eps[t], def_br[t][h] - v)
                atk_result.append(([t] + self.decode_history(h), def_br[t][h] - v))

        def_result = []

        def_pbne_eps = 0.
        for h, v in def_u.items():
            def_pbne_eps = max(def_pbne_eps, atk_br[h] - v)
            def_result.append((self.decode_history(h), atk_br[h] - v))

        if verbose:
            print("PBNE:", atk_pbne_eps, def_pbne_eps)

        atk_eps = [0.] * self.n_types
        initial_state = self.encode_history([])
        for t in range(self.n_types):
            atk_eps[t] += def_br[t][initial_state] - atk_u[t][initial_state]

        def_eps = atk_br[initial_state] - def_u[initial_state]

        if verbose:
            print("BR:", [def_br[t][initial_state] for t in range(self.n_types)], atk_br[initial_state])
            print("Payoff:", [atk_u[t][initial_state] for t in range(self.n_types)], def_u[initial_state])

            print("Overall:", atk_eps, def_eps)

        for_sheet += atk_eps + [def_eps] + [atk_u[t][initial_state] for t in range(self.n_types)] + [def_u[initial_state]]

        if verbose:
            print("\t".join(list(map(str, for_sheet))))

        return ((atk_eps, atk_pbne_eps), (def_eps, def_pbne_eps)), ([atk_u[t][initial_state] for t in range(self.n_types)], def_u[initial_state])

    def get_strategy_profile(self, strategies):
        atk_s, dfd_s = strategies
        atk_p = [atk_s.strategy(self._get_atk_ob(t, self.prior, 0)) for t in range(self.n_types)]
        dfd_p = dfd_s.strategy(self._get_dfd_ob(self.prior, 0))

        return atk_p, dfd_p

    def assess_strategies(self, strategies, verbose=False):
        if self.random_prior and self.n_types == 2:
            for x in range(11):
                self.prior = np.array([x / 10., (10 - x) / 10.])
                self._assess_strategies(strategies, verbose)
        else:
            return self._assess_strategies(strategies, verbose)

        if verbose:
            raise NotImplementedError
        else:
            return []
            # return [[np.sum(np.array(atk_eps) * np.array(self.prior)),
            #          np.sum(atk_pbne_eps * np.array(self.prior))], [def_eps, def_pbne_eps]]

    def get_def_payoff(self, atk_ac, def_ac, prob):
        # ret = 0.
        # for t in range(self.n_types):
        #     ret += prob[t] * self.payoff[t, atk_ac, def_ac, 1]
        ret = self.payoff[0, atk_ac, def_ac, 1]
        return ret

    def get_atk_payoff(self, t, atk_ac, def_ac):
        return self.payoff[t, atk_ac, def_ac, 0]

    def convert_to_atk_init_ob(self, t, prior=None):
        ob = np.zeros(shape=self.n_types)
        ob[t] = 1.0
        if prior is None:
            prior = self.prior
        return np.concatenate([prior, ob])

    def convert_to_def_init_ob(self, prior=None):
        if prior is None:
            prior = self.prior
        return prior

    def convert_to_info_ob(self, history):
        r = len(history)
        ob = np.zeros(shape=(r + 1, 1 + self.n_slots))
        # print(r)
        ob[0][0] = self.n_rounds
        for i in range(r):
            ob[i + 1][0] = self.n_rounds - i - 1
            ob[i + 1][history[i] + 1] = 1.0

        return ob

    # def assess_strategies(self, strategies):
    #     return self.strategies_assessment.run(strategies[0], strategies[1])

    class _AttackerStrategyExploiter(object):
        def __init__(self, env):
            self.cache = None
            self.strategy = None
            self.n_slots = env.n_slots
            self.n_types = env.n_types
            self.n_rounds = env.n_rounds
            self.prior = env.prior
            self.payoff = env.payoff
            self.beta = env.beta

            self._get_def_payoff = env.get_def_payoff
            self._encode_history = env.encode_history

        def _reset(self):
            self.cache = dict()

        def _recursive(self, history, prior, k):
            encoded = self._encode_history(history)
            if len(history) >= self.n_rounds:
                return 0.0
            if encoded in self.cache:
                return self.cache[encoded]
            else:
                atk_strategy_type = np.zeros(shape=(self.n_slots, self.n_types))
                for t in range(self.n_types):
                    atk_strategy = self.strategy[t][encoded]
                    for i in range(self.n_slots):
                        atk_strategy_type[i][t] += atk_strategy[i] * prior[t]

                max_ret = -1e100
                for def_ac in range(self.n_slots):
                    ret = 0.
                    for atk_ac in range(self.n_slots):
                        p = np.sum(atk_strategy_type[atk_ac])
                        # print("sss", atk_ac, p)
                        prob = atk_strategy_type[atk_ac] / p
                        if p < 1e-5:
                            continue
                        next_history = history + [atk_ac]
                        tmp = self._recursive(next_history, prob, k * self.beta)
                        r = self._get_def_payoff(atk_ac, def_ac, prob) * k + tmp
                        ret += r * p
                    # print(history, def_ac, ret)
                    if ret > max_ret:
                        max_ret = ret
                self.cache[encoded] = max_ret
                # print(history, prior, max_ret)
                return max_ret

        def run(self, attacker_strategy, prior):
            self._reset()
            self.strategy = attacker_strategy
            # self.init_ob = self._convert_to_def_init_ob()
            # print(self.prior)
            self.prior = prior
            self._recursive([], self.prior, 1.0)
            return self.cache

    class _DefenderStrategyExploiter(object):
        def __init__(self, env):
            self.cache = None
            self.strategy = None
            self.n_slots = env.n_slots
            self.n_types = env.n_types
            self.n_rounds = env.n_rounds
            self.prior = env.prior
            self.payoff = env.payoff
            self.beta = env.beta

            self._get_atk_payoff = env.get_atk_payoff
            self._encode_history = env.encode_history

        def _reset(self):
            self.cache = dict()

        def _recursive(self, history, t, k):
            encoded = self._encode_history(history)
            if len(history) >= self.n_rounds:
                return 0.0
            if encoded in self.cache:
                return self.cache[encoded]
            else:
                def_strategy = self.strategy[encoded]

                max_ret = -1e100
                for atk_ac in range(self.n_slots):
                    ret = 0.
                    for def_ac in range(self.n_slots):
                        p = def_strategy[def_ac]
                        next_history = history + [atk_ac]
                        tmp = self._recursive(next_history, t, k * self.beta)
                        r = self._get_atk_payoff(t, atk_ac, def_ac) * k + tmp
                        ret += r * p
                    # print("sss", ret)
                    if ret > max_ret:
                        max_ret = ret
                self.cache[encoded] = max_ret
                return max_ret

        def run(self, defender_strategy):
            self.strategy = defender_strategy
            ret = []
            for t in range(self.n_types):
                self._reset()
                self._recursive([], t, 1.0)
                ret.append(deepcopy(self.cache))
            return ret

    class _DefenderUtilityCalculator(object):
        def __init__(self, env):
            self.cache = None
            self.attacker_strategy = None
            self.defender_strategy = None
            self.n_slots = env.n_slots
            self.n_types = env.n_types
            self.n_rounds = env.n_rounds
            self.prior = env.prior
            self.payoff = env.payoff
            self.beta = env.beta

            self._get_def_payoff = env.get_def_payoff
            self._encode_history = env.encode_history

        def _reset(self):
            self.cache = dict()

        def _recursive(self, history, prior, k):
            encoded = self._encode_history(history)
            if len(history) >= self.n_rounds:
                return 0.0
            if encoded in self.cache:
                return self.cache[encoded]
            else:
                atk_strategy_type = np.zeros(shape=(self.n_slots, self.n_types))

                for t in range(self.n_types):
                    atk_strategy = self.attacker_strategy[t][encoded]
                    for i in range(self.n_slots):
                        atk_strategy_type[i][t] += atk_strategy[i] * prior[t]

                utility = 0.0
                def_strategy = self.defender_strategy[encoded]
                for def_ac in range(self.n_slots):
                    p_def = def_strategy[def_ac]
                    for atk_ac in range(self.n_slots):
                        p_atk = np.sum(atk_strategy_type[atk_ac])
                        if p_atk < 1e-5:
                            continue
                        p_type = atk_strategy_type[atk_ac] / p_atk
                        next_history = history + [atk_ac]
                        tmp = self._recursive(next_history, p_type, k * self.beta)
                        r = self._get_def_payoff(atk_ac, def_ac, p_type) * k + tmp
                        utility += r * p_def * p_atk
                self.cache[encoded] = utility
                return utility

        def run(self, attacker_strategy, defender_strategy, prior):
            self._reset()
            self.attacker_strategy = attacker_strategy
            self.defender_strategy = defender_strategy
            self.prior = prior
            self._recursive([], self.prior, 1.0)
            return self.cache

    class _AttackerUtilityCalculator(object):
        def __init__(self, env):
            self.cache = None
            self.freq = None
            self.attacker_strategy = None
            self.defender_strategy = None
            self.n_slots = env.n_slots
            self.n_types = env.n_types
            self.n_rounds = env.n_rounds
            self.prior = env.prior
            self.payoff = env.payoff
            self.beta = env.beta

            self._get_atk_payoff = env.get_atk_payoff
            self._encode_history = env.encode_history

        def _reset(self):
            self.cache = dict()
            self.freq = dict()

        def _recursive(self, history, t, k):
            encoded = self._encode_history(history)
            if len(history) >= self.n_rounds:
                return 0.0
            if encoded in self.cache:
                return self.cache[encoded]
            else:
                utility = 0.0
                atk_strategy = self.attacker_strategy[t][encoded]
                def_strategy = self.defender_strategy[encoded]
                for def_ac in range(self.n_slots):
                    p_def = def_strategy[def_ac]
                    for atk_ac in range(self.n_slots):
                        p_atk = atk_strategy[atk_ac]
                        next_history = history + [atk_ac]
                        tmp = self._recursive(next_history, t, k * self.beta)
                        r = self._get_atk_payoff(t, atk_ac, def_ac) * k + tmp
                        utility += r * p_def * p_atk
                self.cache[encoded] = utility
                return utility

        def run(self, attacker_strategy, defender_strategy, prior):
            self.attacker_strategy = attacker_strategy
            self.defender_strategy = defender_strategy
            self.prior = prior

            ret = []
            for t in range(self.n_types):
                self._reset()
                self._recursive([], t, 1.0)
                ret.append(deepcopy(self.cache))

            return ret