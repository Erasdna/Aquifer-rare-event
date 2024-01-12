import numpy as np
import matplotlib.pyplot as plt 
import pickle 
import os
from scipy.stats import norm

plt.rcParams.update({
        'font.size': 16,
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{amsmath}'
        })
markers = ["^", "<", ">", "o"]

with open(os.getcwd() + "/experiments/Data/splitting_N=1000.pkl","rb") as file:
    probs, num, init, scales = pickle.load(file)

ref = 1.4493907383710904e-18

labs = ["Linear", "Square", "Exponential"]
fig,ax = plt.subplots()
ax.plot(num,np.ones_like(num)*ref,lw=2,label="Feynman-Kac",color="k")
for i in range(len(scales)):
    mid = np.mean(probs[i,:,:],axis=0)
    ax.plot(num,mid,marker=markers[i], markersize=10, lw=2,label=labs[i])
    CI = norm.ppf(0.975)*np.std(probs[i,:,:],axis=0)/np.sqrt(5)
    ax.fill_between(num,mid + CI, mid - CI,alpha=0.3)
ax.legend()
ax.grid()
ax.set_xlabel(r"Total cocentric circles")
ax.set_ylabel(r"$\mathbb{P}(\tau \leq T)$")
fig.savefig("Figures/Splitting/split.png",bbox_inches='tight')
fig.savefig("Figures/Splitting/split.eps",bbox_inches='tight')

