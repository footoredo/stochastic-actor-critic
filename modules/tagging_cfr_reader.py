import joblib
import numpy as np


def to_index(belief, opp_pos, pro_pos):
    a0 = int(belief[0] * 10 + 0.5)
    a1 = int(opp_pos[0])
    a2 = int(opp_pos[1])
    a3 = int(pro_pos[0])
    a4 = int(pro_pos[1])
    return a0, a1, a2, a3, a4


class Reader(object):
    def __init__(self, filename):
        print("Loading {}".format(filename))
        results = joblib.load("{}.obj".format(filename))
        jobs = joblib.load("{}_jobs.obj".format(filename))

        index = np.zeros((11, 8, 8, 8, 4), dtype=np.int)

        for i, job in enumerate(jobs):
            belief, opp_pos, pro_pos = job
            index[to_index(belief, opp_pos, pro_pos)] = i

        self.index = index
        self.results = results

        print("Completed")

    def access(self, belief, opp_pos, pro_pos):
        return self.results[self.index[to_index(belief, opp_pos, pro_pos)]]


class ApproximatedReader(Reader):
    def __init__(self, filename):
        super(ApproximatedReader, self).__init__(filename)
        self.multiplier = 10

    def access(self, belief, opp_pos, pro_pos):
        x = belief[0]
        li = int(x * self.multiplier)
        ri = li + 1
        lx = li / self.multiplier
        rx = ri / self.multiplier

        la = super(ApproximatedReader, self).access([lx], opp_pos, pro_pos)
        if ri > self.multiplier:
            return la
        else:
            ra = super(ApproximatedReader, self).access([rx], opp_pos, pro_pos)
            ret = []
            for i in range(len(la)):
                ret.append(la[i] * (rx - x) / (rx - lx) + ra[i] * (x - lx) / (rx - lx))
            return ret


def main():
    reader = Reader("tagging-8_cfr")
    while True:
        inputs = input().split()
        b = float(inputs[0])
        ox, oy, px, py = map(float, inputs[1:])
        print(reader.access([b], [ox, oy], [px, py]))


if __name__ == "__main__":
    main()
