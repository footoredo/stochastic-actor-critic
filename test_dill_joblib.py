import dill
import joblib
import time
import numpy as np
from tagging_cfr import save_buffer, load_buffer


def test_dill(a):
    st = time.time()
    dill.dump(a, open("test_dill", "wb"))
    print(time.time() - st)


def test_joblib(a):
    st = time.time()
    joblib.dump(a, "test_joblib")
    print(time.time() - st)


def test_numpy(a):
    print(type(a))
    st = time.time()
    np.save("test_numpy", a)
    print(time.time() - st)


def main():
    # a = joblib.load("pro_v_buffer.obj")
    # a = np.array(a["input"])
    a = dict(input=np.random.randn(1000000, 50),
             output=np.random.randn(1000000))
    test_dill(a)
    test_dill(a)
    print("Testing dill")
    test_dill(a)
    print("Testing joblib")
    test_joblib(a)
    print("Testing numpy")
    test_numpy(a)


"""
Testing dill
0.46314501762390137
Testing joblib
0.38428735733032227
Testing numpy
<class 'numpy.ndarray'>
0.1920456886291504
"""


def resave(filename):
    buffer = load_buffer(filename)
    save_buffer(buffer, filename)


if __name__ == "__main__":
    # main()
    resave("pro_v_buffer.obj")
    resave("opp_v_buffer.obj")