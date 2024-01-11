import numpy as np
import matplotlib.pyplot as plt 
import pickle 
import os

plt.rcParams.update({
        'font.size': 16,
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{amsmath}'
        })
markers = ["^", "<", ">", "o"]

with open(os.getcwd() + "/experiments/Data/splitting.pkl") as file:
    probs, num, init, scales = pickle.load(file)


labs = ["Linear", "Square", "Exponential"]
fig,ax = plt.subplots()
for i in range(len(scales)):
    ax.plot(num,np.mean(probs[i,:,:],axis=0),marker=markers[i], markersize=10, lw=2,label=labs[i])
ax.legend()
ax.grid()
ax.set_xlabel(r"Total cocentric circles")
ax.set_ylabel(r"\mathbb{P}(\tau \leq T)")
fig.savefig("Figures/Splitting/split.png",bbox_inches='tight')
fig.savefig("Figures/Splitting/split.eps",bbox_inches='tight')

