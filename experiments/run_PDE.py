import sys
import os

sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from PDE import PDE_solve
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
    [-3.0,4.0]
])
u=PDE_solve(N=100,points=points,dof=200,cutoff=10.0)

ff = lambda x,y: grad(u)(x,y)/(1+u(x,y))
print(ff(3.0,3.0))
print(ff(2.5,2.5))

plot(u)
plt.savefig("Figures/PDE_example.png")
plt.savefig("Figures/PDE_example.eps")
plt.show()
