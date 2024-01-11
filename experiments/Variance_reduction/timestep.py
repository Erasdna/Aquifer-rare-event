import numpy as np
import matplotlib.pyplot as plt 

import os
import sys
sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from Variance_simulator import phi_log,phi_sq,phi,full
from SDE_simulator import SDE
from PDE_simulator import PDE

plt.rcParams.update({
        'font.size': 16,
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{amsmath}'
        })
markers = ["^", "<", ">", "o"]

resolution = 100
cutoff = 20.0
points = []

init = np.array([-2.5,2.5],dtype=float)

alpha = np.array([0.6,0.6,1.0])
n=10
N0 = 10
N = N0*2**np.arange(0,n,dtype=int)

engines = [phi(),phi_sq(),phi_log()]
results = np.zeros((len(engines),len(N),2))
base = np.zeros((len(N),2))

pde_engine = PDE()
base_engine = SDE()

for j,nn in enumerate(N):
    print(nn)
    base[j] = base_engine.solve(init=init,N=nn)
    pre_run = pde_engine.solve(N=nn,resolution=resolution,cutoff=cutoff,points=points,all_timesteps=True)
    for i,e in enumerate(engines):
        print(e)
        results[i,j,:] = e.solve(init=init,N=nn,resolution=resolution,alpha=alpha[i],pre_run=pre_run)
        print(results[i,j,:])


fig,ax = plt.subplots()
ax.semilogx(10/N,base[:,1],lw=2,linestyle="--",color="k",label="Baseline SDE")
labs = [r"$ \nabla \varphi $", r"$ \nabla \varphi^2 $", r"$ \nabla \log(1 + \varphi) $"]
for i,_ in enumerate(engines):
    ax.semilogx(10/N,results[i,:,1],lw=2,marker=markers[i],markersize=10,label=labs[i])
ax.legend()
ax.grid()
ax.set_xlabel(r"$\Delta t$")
ax.set_ylabel(r"$\sigma$")
fig.savefig("Figures/Variance_reduction/variance_timestep.png",bbox_inches='tight')
fig.savefig("Figures/Variance_reduction/variance_timestep.eps",bbox_inches='tight')




