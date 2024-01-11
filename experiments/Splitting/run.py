import numpy as np
import pickle

import sys
import os

sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from Splitting_simulator import SDE_splitting


scales = ['lin','sq','exp']
init = np.array([7.0,7.0],dtype=float)

n1,n2 = 5,10
probs = np.zeros((3,n2,n1))


num = np.linspace(10,100,n2,dtype=int)
for s in range(3):
    for j,n in enumerate(num):
        for i in range(n1):
            solver = SDE_splitting(seed=50+i)
            ret = solver.solve(init,np.linspace(1.0,np.linalg.norm(init)-0.5, n+1),T=10.0,N=1000,reps=1000, scale=scales[s])
            probs[s,j,i] = ret[0]

with open(os.getcwd() + "/experiments/Data/splitting.pkl","wb") as file:
    pickle.dump([probs, num, init, scales],file)

print(np.mean(probs,axis=-1))
print(np.std(probs,axis=-1))