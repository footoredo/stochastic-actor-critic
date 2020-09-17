import joblib
import numpy as np


def main():
    results = joblib.load("tagging-8_cfr.obj")
    n_sample = 7
    n = 8

    def query(b, ox, oy, px, py):
        i = py + px * n + oy * n * n + ox * n * n * n + b * n * n * n * n
        return results[i]

    while True:
        b, ox, oy, px, py = map(int, input().split())
        print(query(b, ox, oy, px, py))


if __name__ == "__main__":
    main()
