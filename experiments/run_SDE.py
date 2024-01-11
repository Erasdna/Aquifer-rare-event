import sys
import os

sys.path.append(os.path.abspath(os.getcwd() + "/src/"))
from SDE_simulator import SDE
import numpy as np 
from scipy.stats import norm


init = np.array([
        [1.2,1.1],
        [2.5,2.5],
        [3.0,4.0],
    ],
    dtype=float
    )

rng = np.random.default_rng(seed=55)
reps=10000
N=1000

solver = SDE()
for pos in init:
    print(pos)
    prob = solver.solve(pos,N=N)
    mean = np.mean(prob)
    std = np.std(prob)
    c = norm.ppf(0.975)
    print("Probability: ", mean*100)
    print("95th percentile CI: [", (mean - c*std/np.sqrt(reps))*100, ", ", (mean + c*std/np.sqrt(reps))*100, "]")

solver2 = SDE(sigma=0.25,Q=-7.0)
for pos in init:
    prob = solver2.solve(pos,N=N)
    mean = np.mean(prob)
    std = np.std(prob)
    c = norm.ppf(0.975)
    print("Probability: ", mean*100)
    print("95th percentile CI: [", (mean - c*std/np.sqrt(reps))*100, ", ", (mean + c*std/np.sqrt(reps))*100, "]")

