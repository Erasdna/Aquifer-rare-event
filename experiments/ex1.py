import sys
import os

sys.path.append(os.path.abspath(os.getcwd() + "/src/"))
from MC_simulation import MC
import numpy as np 
import matplotlib.pyplot as plt

Q=-10.0
R=1.0
sigma = 0.5
T=10
N=1000

init = np.array([2.5,2.5])

def u(pos):
    steady = np.array([1,0])
    perturbation = pos/np.linalg.norm(pos)**2
    return steady + (Q/(2*np.pi))*perturbation

stop,fin = MC(init,u,sigma=sigma,N=N)

t,count = np.unique(stop,return_counts=True)

fig,ax = plt.subplots()
ax.scatter(t[1:],count[1:])
plt.show()

fig2,ax2 = plt.subplots()
ax2.scatter(fin)
plt.show()