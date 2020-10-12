import numpy as np
from env.security_game import SecurityGame


def main(env_seed, n_round):
    env = SecurityGame(2, 2, [0.5, 0.5], n_round, seed=env_seed)

    def atk_random_strategy(t, h):
        return np.array([0.5, 0.5])

    def dfd_random_strategy(h):
        return np.array([0.5, 0.5])

    env.assess_strategies((atk_random_strategy, dfd_random_strategy), verbose=True)


if __name__ == "__main__":
    main(9418, 10)
