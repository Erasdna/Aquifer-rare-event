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

from PDE import PDE_solve

steps = np.array([10.0,15.0,20.0,25.0,30.0,35.0])
#steps = np.array([10.0,15.0,20.0])
N = 100

points = np.array([[-1.2,1.1],[-2.5,2.5],[-3.0,4.0]],dtype=float)
res = np.zeros((len(steps),len(points)))
for i,s in enumerate(steps):
    print(s)
    u=PDE_solve(N=N,cutoff=s,points=points)
    for j,p in enumerate(points):
        res[i,j]=u(p[0],p[1])
    
fig,ax=plt.subplots()
for j,p in enumerate(points):
    ax.semilogy(steps[:-1],np.abs(res[:-1,j]-res[-1,j]),lw=2,label="$\mathbf{X}_0=(" + str(p[0]) + ", " + str(p[1]) + ")$",marker=markers[j],markersize=10)
ax.legend()
ax.set_xlabel("Cutoff radius")
ax.set_ylabel("Error")
ax.grid()
fig.savefig("Figures/Convergence/PDE_cutoff.png",bbox_inches='tight')
fig.savefig("Figures/Convergence/PDE_cutoff.eps",bbox_inches='tight')
#plt.show()