import numpy as np
import pickle
from tqdm import tqdm
from scipy.stats import norm

import sys
import os

sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from Splitting_simulator import SDE_splitting


scales = ["sq", "exp"]
init = np.array(
    [
        [7.0,7.0],
        [2.5, 2.5],
        [3.0, 4.0],
    ],
    dtype=float,
)

n = 5
num = 20
probs = np.zeros((len(init), len(scales), n))
for k, start in enumerate(init):
    for s in range(3):
        print(scales[s])
        for i in tqdm(range(n)):
            solver = SDE_splitting(seed=50 + i)
            ret = solver.solve(
                start,
                num,
                split="log",
                T=10.0,
                N=1000,
                reps=1000,
                scale=scales[s],
                verbose=False,
            )
            probs[k, s, i] = ret[0]
        print("Scale: ", scales[s])
        print("Starting position: ", start)
        mu = 100 * np.mean(probs[k, s, :])
        print("Mean: ", mu)
        CI = 100 * norm.ppf(0.975) * np.std(probs[k, s, :]) / np.sqrt(n)
        print("CI: [", mu - CI, ", ", mu + CI, "]")
