import numpy as np
import matplotlib.pyplot as plt 
plt.rcParams.update({
        'font.size': 16,
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{amsmath}'
        })
markers = ["^", "<", ">", "o"]

import os
import sys
sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from SDE import SDE_solve

n = 10
steps = 2**np.arange(0,n)
N0 = 20
points = np.array([[-1.2,1.1],[-2.5,2.5],[-3.0,4.0]],dtype=float)

mu = np.zeros((len(steps),len(points)))

rng = np.random.default_rng(seed=55)
reps = 10000
path = rng.normal(loc=0.0,scale=1.0,size=(reps,steps[-1]*N0,2))
for i,s in enumerate(steps):
    print(s)
    for j,p in enumerate(points):
        stop = SDE_solve(p,path[:,::s,:])
        mu[n-i-1,j]= len(stop[stop>=0])/reps

print(mu)
fig,ax=plt.subplots()
for j,p in enumerate(points):
    ax.loglog(1/(N0*steps[:-1]),np.abs(mu[:-1,j]-mu[-1,j]),lw=2,label="$\mathbf{X}_0=(" + str(p[0]) + ", " + str(p[1]) + ")$",marker=markers[j],markersize=10)
ax.legend()
ax.set_xlabel("$\Delta t$")
ax.set_ylabel("Error")
ax.grid()
fig.savefig("Figures/Convergence/SDE_timestep.png",bbox_inches='tight')
fig.savefig("Figures/Convergence/SDE_timestep.eps",bbox_inches='tight')
#plt.show()