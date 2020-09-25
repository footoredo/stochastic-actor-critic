import joblib
import numpy as np
import random


def split(original_file, ratio, file1, file2):
    with open(original_file, "r") as f:
        lines = f.readlines()
    n = len(lines)
    s = int(ratio * n)
    random.shuffle(lines)
    with open(file1, "w") as f:
        f.writelines(lines[:s])
    with open(file2, "w") as f:
        f.writelines(lines[s:])


def main():
    jobs = joblib.load("tmp-jobs.obj")
    results = joblib.load("tmp-results.obj")

    n = len(jobs)
    print(len(jobs), len(results))
    opp0 = []
    opp1 = []
    pro = []

    for i in range(n):
        job, result = jobs[i], results[i]
        belief, opp_pos, pro_pos = job
        opp_av, pro_av, opp_v, pro_v, opp_cfv, pro_cfv = result
        features = list(np.concatenate([belief[0], opp_pos, pro_pos]))
        str_f = " ".join(map(lambda x: "{}:{}".format(x[0], x[1]), enumerate(features)))
        opp0.append("{} {}\n".format(opp_v[0], str_f))
        opp1.append("{} {}\n".format(opp_v[1], str_f))
        pro.append("{} {}\n".format(pro_v[0], str_f))

    with open("opp0.txt", "w") as f:
        f.writelines(opp0)

    with open("opp1.txt", "w") as f:
        f.writelines(opp1)

    with open("pro.txt", "w") as f:
        f.writelines(pro)


if __name__ == "__main__":
    # main()
    split("pro.txt", 0.9, "pro.txt.train", "pro.txt.test")
