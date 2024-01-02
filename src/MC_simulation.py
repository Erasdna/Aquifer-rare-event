import numpy as np
from tqdm import tqdm

def MC(init,u,sigma,N,R=1,T=10,reps=10000,seed=55):
    dt = T/N
    assert(dt < R**2)
    assert(init.shape==(2,))

    rng = np.random.default_rng(seed=seed)
    stop = np.zeros(reps)
    fin = np.zeros_like(stop)
    for it in tqdm(range(reps)):
        pos = init.copy()
        stop[it],fin[it] = sim(pos,u,sigma,N,R,T,rng,dt)
    print(len(stop[stop>=0])/reps)
    return stop,fin 
        
def sim(pos,u,sigma,N,R,T,rng,dt):
    for t in np.linspace(0,T,N):
            pos += u(pos)*dt + sigma*np.sqrt(dt)*rng.normal(loc=0,scale=1,size=(2,))
            if np.linalg.norm(pos)<= R:
                return t,np.linalg.norm(pos)
    return -1,np.linalg.norm(pos)