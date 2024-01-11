import numpy as np
from tqdm import tqdm
from src.old.PDE import PDE_solve
import matplotlib.pyplot as plt

def SDE_solve_phi(init,seed,reps=10000,sigma=0.25,R=1.0,T=10,N=100,Q=-7.0,resolution=100,cutoff=20.0):
    dt=T/N

    phi = PDE_solve(N,sigma=sigma,R=R,T=T,Q=Q,cutoff=cutoff,dof=resolution,points=[],verbose=False,all_timesteps=True)

    rng = np.random.default_rng(seed=seed)

    #Baseline flow-field
    def u(pos):
        steady = np.array([1.0,0.0])
        perturbation = pos/np.linalg.norm(pos)**2
        return steady + (Q/(2*np.pi))*perturbation
    
    def reduction(pos,it):
        if np.linalg.norm(pos)>= cutoff:
            print(pos)
        dir = np.array(phi[it][1](pos[0],pos[1]))
        fac = 1/(sigma*2*np.pi) #*(Q/(2*np.pi))
        
        mean = fac*dir*np.abs(phi[it][0](pos[0],pos[1])) 
        return mean,rng.normal(loc=mean*dt,scale=np.sqrt(dt),size=(2,))

    stop = np.zeros(reps)
    est = np.zeros(reps)
    for it in tqdm(range(reps)):
        pos = init.copy()
        stop[it],logw = sim(pos,u,reduction,sigma,N,R,T,dt)
        est[it]= (stop[it]>=0)*np.exp(logw)
        print(est[it])
    
    mean = np.mean(est)
    std = np.sqrt(np.mean((est-mean)**2))
    return mean,std

def sim(pos,u,reduction,sigma,N,R,T,dt):
    logw = 0
    history = np.zeros((N,2))
    for i,t in enumerate(np.linspace(0,T,N)):
        history[i,:] = pos
        func,xi = reduction(pos,N-i-1)
        logw += 0.5*dt*np.linalg.norm(func)**2 - func.T @ xi
        pos += u(pos)*dt + sigma*xi
        if np.linalg.norm(pos)<= R:
                #print(logw)
                #fig,ax = plt.subplots()
                #ax.plot(history[:,0],history[:,1])
                #plt.show()
                return t,logw 
    return -1,logw

def SDE_solve_distance():
    raise NotImplementedError