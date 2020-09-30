from utils.utils import *
import time
import os


def parse_args():
    parser = get_parser("run_sec")
    parser.add_argument('--n-samples', type=int, default=20)
    parser.add_argument('--start-round', type=int, default=1)
    parser.add_argument('--cfr', action="store_true", default=False)
    parser.add_argument('--pred', action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    command = "python mmmm.py --env-seed={env_seed} --n-slots={n_slots} --n-rounds={n_rounds} --n-iter={n_iter} " \
              "--n-samples={n_samples} --build --all --save-data" + (" --cfr" if args.cfr else "") + \
              (" --pred" if args.pred else "")
    start_round = args.start_round
    n_rounds = args.n_rounds
    args = vars(args)
    for i in range(start_round - 1, n_rounds):
        print("Round #", i + 1)
        st = time.time()
        args["n_rounds"] = i + 1
        os.system(command.format(**args))
        print("Used time: {}s".format(time.time() - st))


if __name__ == "__main__":
    main()
