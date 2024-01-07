import sys
import os

sys.path.append(os.path.abspath(os.getcwd() + "/src/"))
from SDE import SDE_solve
import numpy as np 
from scipy.stats import norm


init = np.array([
        [1.2,1.1],
        [2.5,2.5],
        [3.0,4.0],
        [-1.2,1.1],
        [-2.5,2.5],
        [-3.0,4.0]
    ],
    dtype=float
    )

rng = np.random.default_rng(seed=55)
reps=10000
N=1000
path = rng.normal(loc=0.0,scale=1.0,size=(reps,N,2))

for pos in init:
    stop = SDE_solve(pos,path)
    mean = len(stop[stop>=0])/reps
    std = np.sqrt(mean*(1-mean))
    c = norm.ppf(0.975)
    print("Probability: ", mean*100)
    print("95th percentile CI: [", (mean - c*std/np.sqrt(reps))*100, ", ", (mean + c*std/np.sqrt(reps))*100, "]")

