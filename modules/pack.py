import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import ts
from scipy.interpolate import CubicSpline, interp1d, Akima1DInterpolator
import numpy as np


class CubicSplinePack(nn.Module):
    def __init__(self, samples):
        super(CubicSplinePack, self).__init__()
        self.n = n = len(samples)
        xs = []
        ys = []
        for i in range(n):
            xs.append(samples[i][0])
            ys.append(samples[i][1])
            # if i % 10 == 0:
            #     print(ys[-1])

        # print(xs, ys)

        self.cs = CubicSpline(xs, ys, extrapolate=True)
        self.dis = 1. / (n - 1)

        # fig, ax = plt.subplots(figsize=(6.5, 4))
        # ax.plot(xs, self.cs(xs))
        # ax.set_xlim(0., 1.)
        # plt.show()

        # print(self.cs(0.8))

    def forward(self, b):
        x = b[0]
        v = torch.zeros_like(x)
        k = 3
        # l = 0
        # r = self.n - 1
        # while l < r - 1:
        #     m = (l + r) // 2
        #     if x < self.cs.x[m]:
        #         r = m
        #     else:
        #         l = m
        l = np.minimum((x.detach().numpy() / self.dis + 1e-5).astype(int), self.n - 2)
        # print(self.n, l, self.cs.x[l])
        bx = x - ts(self.cs.x[l])
        a = torch.ones_like(x)
        for m in reversed(range(k + 1)):
            v += a * ts(self.cs.c[m, l])
            a = a * bx
        return v


class Akima1DPack(nn.Module):
    def __init__(self, samples):
        super(Akima1DPack, self).__init__()
        self.n = n = len(samples)
        xs = []
        ys = []
        for i in range(n):
            xs.append(samples[i][0][0])
            ys.append(samples[i][1][0])

        # print(xs, ys)

        self.cs = Akima1DInterpolator(xs, ys)
        # print(self.cs(0.8))

    def forward(self, b):
        v = torch.tensor(0.)
        k = 3
        x = b[0]
        for i in range(0, self.n - 1):
            if x < self.cs.x[i + 1]:
                bx = x - self.cs.x[i]
                a = torch.tensor(1.)
                for m in reversed(range(k + 1)):
                    v += a.mul(self.cs.c[m, i])
                    a = a.mul(bx)
                break
        return v


class Interp1dPack(nn.Module):
    def __init__(self, samples):
        super(Interp1dPack, self).__init__()
        self.n = n = len(samples)
        xs = []
        ys = []
        for i in range(n):
            xs.append(samples[i][0])
            ys.append(samples[i][1])

        self.xs = np.array(xs)
        self.ys = np.array(ys)
        self.dis = 1. / (n - 1)
        # print(xs, ys)
        # print(self.cs(0.8))

    def forward(self, b):
        # b: [batch, types=2]
        x = b[:, 0]
        i = np.minimum((x.detach().numpy() / self.dis + 1e-5).astype(int), self.n - 2)  # [batch] int
        wa = x - ts(self.xs[i])  # [batch] float
        wb = ts(self.xs[i + 1]) - x  # [batch] float
        w = ts(self.xs[i + 1] - self.xs[i])  # [batch] float
        return wb / w * ts(self.ys[i]) + wa / w * ts(self.ys[i + 1])


class Interp1dPackFast(nn.Module):
    def __init__(self, samples):
        super(Interp1dPackFast, self).__init__()
        self.n = n = len(samples)
        xs = []
        ys = []
        for i in range(n):
            xs.append(samples[i][0])
            y = samples[i][1:]
            if y.shape[0] == 1:
                y = y[0]
            ys.append(y)

        self.xs = np.array(xs, dtype=np.float)
        self.ys = np.array(ys, dtype=np.float)
        self.dis = 1. / (n - 1)
        # print(xs, ys)
        # print(self.cs(0.8))

    def forward(self, b, eps=None):
        x = b[0]
        # xx = x.detach().item()
        # ll = 0
        # rr = self.n - 1
        # while ll < rr - 1:
        #     m = (ll + rr) // 2
        #     if self.xs[m] <= x:
        #         ll = m
        #     else:
        #         rr = m
        # i = ll
        i = min(self.n - 2, int(x.detach().item() / self.dis + 1e-5))
        wa = x - ts(self.xs[i])
        wb = ts(self.xs[i + 1]) - x
        w = ts(self.xs[i + 1] - self.xs[i])
        # print(self.ys[i], self.ys[i + 1])
        return (wb / w * ts(self.ys[i]) + wa / w * ts(self.ys[i + 1])).detach()


class NNPack(nn.Module):
    def __init__(self, samples):
        super(NNPack, self).__init__()
        self.n = n = len(samples)
        xs = []
        ys = []
        for i in range(n):
            xs.append(samples[i][0])
            y = samples[i][1:]
            if y.shape[0] == 1:
                y = y[0]
            ys.append(y)

        self.xs = np.array(xs, dtype=np.float)
        self.ys = np.array(ys, dtype=np.float)
        self.dis = 1. / (n - 1)
        # print(xs, ys)
        # print(self.cs(0.8))

    def forward(self, b, eps=ts(1e-5)):
        x = b[0]
        sv = 0.
        sw = 0.
        for i in range(self.n):
            dis = torch.abs(x - self.xs[i])
            w = ts(1.) / (dis * dis).clamp(min=eps)
            # dis = torch.abs(x - self.xs[i]) - 0.1
            # if dis <= 0:
            #     w = 1. / eps * (1. - dis / (eps / 1e-2))
            # else:
            #     w = 1. / (eps + dis * dis)

            # dis = dis.clamp(min=eps)
            # dis = torch.abs(x - self.xs[i])
            # w = 1. / (eps + dis * dis)
            # w = torch.exp(-(0.2 / eps) * dis * dis)
            # w = max(0., 2 - np.exp(dis))
            sw += w
            sv += w * ts(self.ys[i])
        return sv / sw


class Interp1dPackUltraFast(object):
    def __init__(self, samples):
        self.n = n = len(samples)
        xs = []
        ys = []
        for i in range(n):
            xs.append(samples[i][0])
            y = samples[i][1:]
            if y.shape[0] == 1:
                y = y[0]
            ys.append(y)

        self.xs = np.array(xs)
        self.ys = np.array(ys)
        self.dis = 1. / (n - 1)
        # print(xs, ys)
        # print(self.cs(0.8))

    def __call__(self, b):
        x = b[0]
        i = min(self.n - 2, int(x / self.dis + 1e-5))
        wa = x - ts(self.xs[i])
        wb = ts(self.xs[i + 1]) - x
        w = self.xs[i + 1] - self.xs[i]
        return wb / w * ts(self.ys[i]) + wa / w * ts(self.ys[i + 1])


class DiscretePackFast(nn.Module):
    def __init__(self, samples):
        super(DiscretePackFast, self).__init__()
        self.n = n = len(samples)
        xs = []
        ys = []
        for i in range(n):
            xs.append(samples[i][0])
            y = samples[i][1:]
            if y.shape[0] == 1:
                y = y[0]
            ys.append(y)

        self.xs = np.array(xs, dtype=np.float)
        self.ys = np.array(ys, dtype=np.float)
        self.dis = 1. / (n - 1)
        # print(xs, ys)
        # print(self.cs(0.8))

    def forward(self, b, eps=None):
        x = b[0].detach().item()
        i = min(self.n - 2, int(x / self.dis + 1e-5))
        wa = x - self.xs[i]
        wb = self.xs[i + 1] - x

        if wa < wb:
            return ts(self.ys[i])
        else:
            return ts(self.ys[i + 1])


if __name__ == "__main__":
    pack = Interp1dPack([[0.0, 0.0], [1.0, 2.0]])
    print(pack(ts([[0.1], [0.2], [0.4], [0.3]])))  # tensor([0.2000, 0.4000, 0.8000, 0.6000])
