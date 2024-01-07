import numpy as np
from tqdm import tqdm

def SDE_solve(init,path,sigma=0.25,R=1.0,T=10,Q=-7.0):    
    reps=path.shape[0]
    N=path.shape[1]
    dt = T/N
    assert(dt <= R**2)
    assert(init.shape==(2,))
    def u(pos):
        steady = np.array([1.0,0.0])
        perturbation = pos/np.linalg.norm(pos)**2
        return steady + (Q/(2*np.pi))*perturbation

    stop = np.zeros(reps)
    for it in tqdm(range(reps)):
        pos = init.copy()
        stop[it] = sim(pos,u,sigma,N,R,T,path[it,:,:],dt)
    return stop 
        
def sim(pos,u,sigma,N,R,T,path,dt):
    for i,t in enumerate(np.linspace(0,T,N)):
            pos += u(pos)*dt + sigma*np.sqrt(dt)*path[i,:]
            if np.linalg.norm(pos)<= R:
                return t
    return -1