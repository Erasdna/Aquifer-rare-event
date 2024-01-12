import numpy as np 
from tqdm import tqdm

from Simulator import Simulator
from PDE_simulator import PDE

class SDE(Simulator):
    def __init__(self, sigma=0.5, R=1, Q=-10.0, seed = 55) -> None:
        super().__init__(sigma, R, Q)
        self.seed=seed
        self.rng = np.random.default_rng(seed=self.seed)
    
    def solve(self, init,T=10, N=1000,reps=10000, save_pos=False):
        stop = np.zeros(reps)
        if save_pos:
            histories=[]
        for it in tqdm(range(reps)):
            pos = init.copy()

            if save_pos:
                stop[it],_,_,tmp = self.sim(pos,T,N,save_pos=save_pos)
                histories.append(tmp)
            else:
                stop[it],_,_ = self.sim(pos,T,N)
        if save_pos:
            return stop, histories
        else:   
            return stop

    def flow(self,pos):
        steady = np.array([1.0,0.0])
        perturbation = pos/np.linalg.norm(pos)**2
        return steady + (self.Q/(2*np.pi))*perturbation

    def path(self,pos,it,dt):
        shift = self.phi(pos,it)
        step = self.rng.normal(loc=shift*dt,scale=np.sqrt(dt),size=(2,))
        log_weight = 0.5*dt*np.linalg.norm(shift)**2 - shift.T @ step 
        return step,log_weight

    def phi(self,pos,it):
        return np.zeros(2)

    def sim(self,pos,T,N, t0=0.0, condition=None, save_pos=False):
        if save_pos:
            history = []

        if condition is not None:
            r = condition
        else:
            r = self.R 

        dt=T/N
        n = int(N - t0/dt)

        logw = 0
        for i,t in enumerate(np.linspace(t0,T,n)):
            if save_pos:
                history.append(pos.copy())
            if np.linalg.norm(pos)<= r or np.linalg.norm(pos)<= self.R:
                if save_pos:
                    return 1.0*np.exp(logw),t,pos, history
                else:
                    return 1.0*np.exp(logw),t,pos  
            
            xi, tmp = self.path(pos,N-i-1,dt)
            logw += tmp 
            pos += self.flow(pos)*dt + self.sigma*xi
        if save_pos:
            return 0.0,T,pos,history
        else:
            return 0.0,T,pos

class SDE_path(SDE):
    def solve(self, init, preloaded_path, T=10, N=1000, reps=10000):
        self.preloaded_path=preloaded_path
        stop = np.zeros(reps)
        for it in tqdm(range(reps)):
            pos = init.copy()
            self.current_path = self.preloaded_path[it,:,:]
            stop[it],_,_ = self.sim(pos,T,N)
        
        return np.mean(stop), np.std(stop)
    def path(self, pos, it, dt):
        return self.current_path[it,:],0.0
