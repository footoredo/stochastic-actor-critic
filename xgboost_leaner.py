import xgboost as xgb
import numpy as np


def main(name):
    dtrain = xgb.DMatrix(name + ".train")
    dtest = xgb.DMatrix(name + ".test")
    param = {
        "booster": "gbtree",
        "objective": "reg:squarederror",
        "eta": 1.0,
        "gamma": 1.0,
        "min_child_weight": 1,
        "max_depth": 10
    }
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, 10, evallist)
    bst.save_model(name + ".model")

    while True:
        data = xgb.DMatrix(np.array([list(map(float, input().split()))]))
        print(bst.predict(data))


if __name__ == "__main__":
    main("pro.txt")
