import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm
import os
import pickle

plt.rcParams.update({
        'font.size': 16,
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{amsmath}'
        })
markers = ["^", "<", ">", "o"]


with open(os.getcwd() + "/experiments/Data/Variance_run_data_N=100.pkl", "rb") as file:
    ref,base,div,results = pickle.load(file)

start = 500
x = np.arange(results.shape[1])

running_base = np.zeros_like(base)
running_std_base = np.zeros_like(running_base)
running_mean = np.zeros_like(results)
running_std = np.zeros_like(running_mean)
for i in range(results.shape[1]):
    running_base[i] = np.mean(base[:i+1])
    running_std_base[i] = np.std(base[:i+1])
    running_mean[:,i] = np.mean(results[:,:i+1],axis=1)
    running_std[:,i] = np.std(results[:,:i+1],axis=1)

print(ref)
print(running_base[-1])
print(running_mean[:,-1])

fig,ax = plt.subplots()
#ax.semilogx(x[start:],ref*np.ones_like(x[start:]),lw=2,linestyle="-",color="k", label="Feynman-Kac")
ax.semilogx(x[start:],np.abs(running_base[start:]-ref)/ref,lw=2,linestyle="--",color="k",label="Baseline SDE")
#ax.fill_between(x[start:], running_base[start:] + running_std_base[start:],running_base[start:] - running_std_base[start:],alpha=0.3)
labs = [r"$\tilde{\mathbf{u}}_1$", r"$\tilde{\mathbf{u}}_2$", r"$\tilde{\mathbf{u}}_3$"]
for i in range(results.shape[0]):
    ax.semilogx(x[start:],np.abs(running_mean[i,start:]-ref)/ref,lw=2,label=labs[i])
    #ax.fill_between(x[start:], running_mean[i,start:] + running_std[i,start:],running_mean[i,start:] - running_std[i,start:],alpha=0.5)
ax.legend()
ax.grid()
ax.set_xlabel(r"MC iterations")
ax.set_ylabel(r"Relative error")
fig.savefig("Figures/Variance_reduction/variance.png",bbox_inches='tight')
fig.savefig("Figures/Variance_reduction/variance.eps",bbox_inches='tight')

fig,ax = plt.subplots()
ax.semilogx(x[start:],running_std_base[start:],lw=2,linestyle="--",color="k",label="Baseline SDE")
print(running_std_base[-1])
#ax.fill_between(x[start:], running_base[start:] + running_CI_base[start:],running_base[start:] - running_CI_base[start:],alpha=0.3)
for i in range(results.shape[0]):
    ax.semilogx(x[start:],running_std[i,start:],lw=2,label=labs[i])
    print(running_std[i,-1])
    #ax.fill_between(x[start:], running_mean[i,start:] + running_CI[i,start:],running_mean[i,start:] - running_CI[i,start:],alpha=0.5)
ax.legend()
ax.grid()
ax.set_xlabel(r"MC iterations")
ax.set_ylabel(r"Standard deviation")
fig.savefig("Figures/Variance_reduction/variance_CI.png",bbox_inches='tight')
fig.savefig("Figures/Variance_reduction/variance_CI.eps",bbox_inches='tight')