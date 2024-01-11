import numpy as np
import matplotlib.pyplot as plt 

import os
import sys
sys.path.append(os.path.abspath(os.getcwd() + "/src/"))

from Variance_simulator import phi_log, dynamic_Q
from SDE_simulator import SDE

init = np.array([2.5,2.5],dtype=float)
engine_base = SDE()
engine = phi_log()
prob = engine.solve(init=init,N=1000,resolution=200,alpha=1.0)
print(prob)

print(engine_base.solve(init,N=1000))