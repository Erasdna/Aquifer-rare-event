import numpy as np
import matplotlib.pyplot as plt 
import pickle

import os
import sys
sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from Variance_simulator import dynamic_Q, dynamic_steady, dynamic_combo
from SDE_simulator import SDE
from PDE_simulator import PDE



plt.rcParams.update({
        'font.size': 16,
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{amsmath}'
        })
markers = ["^", "<", ">", "o"]

resolution = 200
cutoff = 20.0
points = []

init = np.array([2.5,2.5],dtype=float)

reps = 10000
N = 100

engine = SDE()
stops,histories = engine.solve(init,reps=reps,N=N,save_pos=True)

#Plot stopping times
p1 = plt.Circle((0,0),1,fill=False, lw=2)

fig,ax = plt.subplots(subplot_kw={'projection': 'polar'})
t1 = np.linspace(0,2*np.pi,1000)
ax.plot(t1,np.ones_like(t1),color="k", label = "Well")

for hist in histories[50:100]:
    arr = np.array(hist)
    r = np.linalg.norm(arr,axis=1)
    theta = np.arctan(arr[:,1]/arr[:,0])
    ax.plot(theta,r,color="k")
ax.legend()
ax.set_rmax(10)
fig.savefig("Figures/Variance_reduction/base.png",bbox_inches='tight')
fig.savefig("Figures/Variance_reduction/base.eps",bbox_inches='tight')
plt.show()




