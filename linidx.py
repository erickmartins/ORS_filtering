import numpy as np


def linidx_take(val_arr, z_indices):

    # Get number of columns and rows in values array
    _, nC, nR = val_arr.shape

    # Get linear indices and thus extract elements with np.take
    idx = nC * nR * z_indices + nR * np.arange(nR)[:, None] + np.arange(nC)
    return np.take(val_arr, idx)  # Or val_arr.ravel()[idx]


def choose_based(val_arr, z_indices):
    return z_indices.choose(val_arr)


if __name__ == "__main__":
    a = np.random.rand(31, 1000, 1000)
    b = np.random.randint(31, size=(1000, 1000))
    # c = b.choose(a)
    d = linidx_take(a, b)

    # print(c - d)
