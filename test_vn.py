from utils.tools import load_vn


if __name__ == "__main__":
    # load_vn("data/sec-cfr-8932-2-2-2-10-10000.atk_vn.obj", "nn")
    load_vn("data/sec-cfr-5410-2-2-2-9-10000.atk_vn.obj", interp="linear_fast", verbose=True)
