import numpy as np
import pickle
from tqdm import tqdm

import sys
import os

sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from Splitting_simulator import SDE_splitting

if __name__ == "__main__":
    split = sys.argv[1]
    scales = ["sq", "exp"]
    init = np.array([7.0, 7.0], dtype=float)

    n1, n2 = 5, 5
    probs = np.zeros((3, n2, n1))

    num = np.linspace(10, 50, n2, dtype=int)
    for s in range(2):
        print(scales[s])
        for j, n in enumerate(num):
            print(n)
            for i in tqdm(range(n1)):
                solver = SDE_splitting(seed=50 + i)
                ret = solver.solve(
                    init, n, split=split, T=10.0, N=1000, reps=1000, scale=scales[s]
                )
                probs[s, j, i] = ret[0]
            print(np.mean(probs[s, j, :]))

    with open(
        os.getcwd() + "/experiments/Data/splitting_" + split + ".pkl", "wb"
    ) as file:
        pickle.dump([probs, num, init, scales], file)

    print(np.mean(probs, axis=-1))
    print(np.std(probs, axis=-1))
