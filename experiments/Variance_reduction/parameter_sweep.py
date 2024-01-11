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

N=1000
resolution = 100
cutoff = 20.0
points = []

init = np.array([2.5,2.5],dtype=float)

alpha = np.linspace(0.25,1.25,9)
engines = [phi(),phi_sq(),phi_log()]
results = np.zeros((len(engines),len(alpha),2))

pde_engine = PDE()
pre_run = pde_engine.solve(N=N,resolution=resolution,cutoff=cutoff,points=points,all_timesteps=True)
for i,e in enumerate(engines):
    print(e)
    for j,a in enumerate(alpha):
        print(a)
        results[i,j,:] = e.solve(init=init,N=N,resolution=resolution,alpha=a,pre_run=pre_run)
        print(results[i,j,:])
base_engine = SDE()
base = base_engine.solve(init=init,N=N)

fig,ax = plt.subplots()
ax.plot(alpha,np.ones_like(alpha)*base[1],lw=2,linestyle="--",color="k",label="Baseline SDE")
labs = [r"$ \nabla \varphi $", r"$ \nabla \varphi^2 $", r"$ \nabla \log(1 + \varphi) $"]
#labs = ["phi","phi squared", "log"]
for i,_ in enumerate(engines):
    ax.plot(alpha,results[i,:,1],lw=2,marker=markers[i],markersize=10,label=labs[i])
ax.legend()
ax.grid()
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\sigma$")
fig.savefig("Figures/Variance_reduction/alpha_sweep_05.png",bbox_inches='tight')
fig.savefig("Figures/Variance_reduction/alpha_sweep_05.eps",bbox_inches='tight')




