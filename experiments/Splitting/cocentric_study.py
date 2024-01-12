import numpy as np
import pickle
from tqdm import tqdm

import sys
import os

sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from Splitting_simulator import SDE_splitting


scales = ['sq','exp','lin']
init = np.array([7.0,7.0],dtype=float)

n1,n2 = 5,1
probs = np.zeros((3,n2,n1))


num = np.linspace(10,50,n2,dtype=int)
for s in range(3):
    print(scales[s])
    for j,n in enumerate(num):
        print(n)
        for i in tqdm(range(n1)):
            solver = SDE_splitting(seed=50+i)
            ret = solver.solve(init,np.linspace(1.0,np.linalg.norm(init)-0.2, n+1),T=10.0,N=1000,reps=1000, scale=scales[s])
            probs[s,j,i] = ret[0]
        print(np.mean(probs[s,j,:]))

with open(os.getcwd() + "/experiments/Data/splitting_2_5.pkl","wb") as file:
    pickle.dump([probs, num, init, scales],file)

print(np.mean(probs,axis=-1))
print(np.std(probs,axis=-1))