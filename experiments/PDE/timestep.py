import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 16,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts} \usepackage{amsmath}",
    }
)
markers = ["^", "<", ">", "o"]

import os
import sys

sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from PDE_simulator import PDE

n = 7
steps = 2 ** np.arange(0, n)
N0 = 10

solver = PDE()
points = np.array([[1.2, 1.1], [2.5, 2.5], [3.0, 4.0]], dtype=float)
res = np.zeros((len(steps), len(points)))
for i, s in enumerate(steps):
    print(s)
    u = solver.solve(N=N0 * s, resolution=100, points=points)
    for j, p in enumerate(points):
        res[i, j] = u(p[0], p[1])

fig, ax = plt.subplots()
for j, p in enumerate(points):
    ax.loglog(
        1 / (N0 * steps[:-1]),
        np.abs(res[:-1, j] - res[-1, j]),
        lw=2,
        label="$\mathbf{X}_0=(" + str(p[0]) + ", " + str(p[1]) + ")$",
        marker=markers[j],
        markersize=10,
    )
ax.legend()
ax.set_xlabel("$\Delta t$")
ax.set_ylabel("Error")
ax.grid()
fig.savefig("Figures/Convergence/PDE_timestep.png", bbox_inches="tight")
fig.savefig("Figures/Convergence/PDE_timestep.eps", bbox_inches="tight")
# plt.show()
