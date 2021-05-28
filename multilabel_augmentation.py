from numba import njit, jit
import numpy as np
from rich.console import Console
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool
from itertools import repeat
from tqdm import trange


def main():


    data = pd.read_csv("/home/talha/Downloads/test.csv", skipinitialspace=True)

    _labels = data.iloc[:,1:-1].values

    N = 20
    C = 15

    Console().rule(title="Augmenting", style="red on black", align="center", characters="|")

    # _labels = np.random.choice([0, 1], size=(N, C))

    Console().print(_labels)

    counts = np.sum(_labels, axis=0)
    Console().rule(title="initial")
    # Console().print(_labels)
    Console().print(counts)


    @njit
    def cost(ref, counts):
        cost = np.sum(ref - counts)
        return cost


    def estim_counts(_labels):
        upd_count = []
        IDX = []

        # upd_c = np.zeros((len(_labels), len(_labels[0])))
        # IDX_ = np.zeros((len(_labels),))

        for I, k in enumerate(_labels):
            #     k ----> row
            dummy = _labels

            dummy = np.vstack((dummy, k.reshape(1, -1)))
            upd_count.append(np.sum(dummy, axis=0))
            IDX.append(I)
            # upd_c[I, :] = np.sum(dummy, axis=0)
            # IDX_[I] = I

        # print(upd_count)
        # print(IDX)
        return upd_count, IDX


    c = []
    for kk in trange(1000, desc="ITER", leave=False):
        # find the largest of all counts
        # counts = np.sum(_labels, axis=0)
        # find shich class is the max counts
        idx = np.argmax(counts)
        # val. to be set  ref.
        ref = counts[idx]

        # defining a cost function

        prev_cost = cost(ref, counts)
        # decide which row to augment
        # iterate over the rows of labels , find counts after augmentation and see which has the largest cost
        # upd_count = []
        # IDX = []
        #
        # for I, k in enumerate(_labels):
        #     #     k ----> row
        #     dummy = _labels
        #     dummy = np.vstack((dummy, k))
        #     upd_count.append(np.sum(dummy, axis=0))
        #     IDX.append(I)
        upd_count, IDX = estim_counts(_labels)

        # upd_count = upd_count.tolist()
        # IDX = IDX.tolist()

        new_costs = [cost(ref, i) for i in upd_count]
        min_idx = np.argmin(new_costs)

        # actually do aumentation
        _labels = np.vstack((_labels, _labels[IDX[min_idx]].reshape(1, -1)))
        counts = upd_count[min_idx]

        # print(f"{str(kk):<15} {counts}")
        c.append(counts)
        # print(prev_cost)

        if prev_cost >= 1000:
            pass

    Console().rule(title="End")
    # Console().print(_labels)
    Console().print(np.sum(_labels, axis=0))
    # c = np.asarray(c)
    #
    # for k in range(c.shape[1]):
    #     plt.scatter(x=list(range(100)), y=c[:,k], s=12, c="r")
    #     # plt.plot(c)
    # plt.show()


    pd.DataFrame(data=_labels).to_csv("augmented_labels.csv", index=False,header=False)

if __name__ == '__main__':
    main()
