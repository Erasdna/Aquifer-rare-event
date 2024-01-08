import numpy as np
import matplotlib.pyplot as plt 

import os
import sys
sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from SDE_simulator import SDE,PDE_importance_sampler,PDE_importance_sampler_sq, Distance_sampler

init = np.array([-2.5,2.5],dtype=float)
engine_base = SDE()
engine = PDE_importance_sampler()
prob = engine.solve(init=init,N=1000,resolution=200)
print(prob)

print(engine_base.solve(init,N=100))