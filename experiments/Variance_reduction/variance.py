import numpy as np
import matplotlib.pyplot as plt
import pickle

import os
import sys

sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from Variance_simulator import dynamic_Q, dynamic_steady, dynamic_combo
from SDE_simulator import SDE
from PDE_simulator import PDE


resolution = 200
cutoff = 20.0
points = []

init = np.array([2.5, 2.5], dtype=float)

reps = 100000
N = 100

engines = [dynamic_steady(Q=-10.0), dynamic_Q(Q=-10.0), dynamic_combo(Q=-10.0)]
results = np.zeros((len(engines), reps))
div = np.arange(1, reps + 1)
running_mean = np.zeros_like(results)
running_CI = np.zeros_like(results)

pde_engine = PDE(Q=-10.0)
base_engine = SDE(Q=-10.0)
base = base_engine.solve(init=init, N=N, reps=reps)

pre_run = pde_engine.solve(
    N=N,
    resolution=resolution,
    cutoff=cutoff,
    points=points,
    all_timesteps=True,
    return_grad=False,
)

ref = pre_run[-1][0](init[0], init[1])
for i, e in enumerate(engines):
    print(e)
    results[i, :] = e.solve(
        init=init, N=N, reps=reps, resolution=resolution, pre_run=pre_run
    )


with open(os.getcwd() + "/experiments/Data/Variance_run_data_N=100.pkl", "wb") as file:
    pickle.dump([ref, base, div, results], file)
