import sys
import os

sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from PDE_simulator import PDE
from dolfin.common.plotting import plot
import matplotlib.pyplot as plt
import numpy as np
from fenics import *

points = np.array([
    [1.2,1.1],
    [2.5,2.5],
    [3.0,4.0],
    [-1.2,1.1],
    [-2.5,2.5],
    [-3.0,4.0],
    [7.0,7.0],
    [-7.0,7.0]
])
solver = PDE(sigma=0.25,Q=-7.0)
u=solver.solve(N=1000,points=points,resolution=300,cutoff=20.0,T=10.0)
print("DoF: ", len(u.vector()))

plot(u)
plt.savefig("Figures/PDE_example.png")
plt.savefig("Figures/PDE_example.eps")
plt.show()
