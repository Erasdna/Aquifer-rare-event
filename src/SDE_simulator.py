import numpy as np 
from tqdm import tqdm

from Simulator import Simulator
from PDE_simulator import PDE

class SDE(Simulator):
    def __init__(self, sigma=0.25, R=1, Q=-7, seed = 55) -> None:
        super().__init__(sigma, R, Q)
        self.seed=seed
        self.rng = np.random.default_rng(seed=self.seed)
    
    def solve(self, init,T=10, N=1000,reps=10000):
        stop = np.zeros(reps)
        for it in tqdm(range(reps)):
            pos = init.copy()
            stop[it] = self.sim(pos,T,N)
        
        return np.mean(stop), np.std(stop)

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

    def sim(self,pos,T,N):
        dt=T/N
        logw = 0
        for i,_ in enumerate(np.linspace(0,T,N)):
            xi, tmp = self.path(pos,N-i-1,dt)
            logw += tmp 
            pos += self.flow(pos)*dt + self.sigma*xi
            if np.linalg.norm(pos)<= self.R:
                return 1.0*np.exp(logw)
        return 0.0

class PDE_importance_sampler(SDE):
    def solve(self, init, T=10, N=1000, reps=10000, resolution=100, cutoff=20.0):
        pde_solver = PDE(self.sigma,self.R,self.Q)
        self.importance = pde_solver.solve(T=T,N=N,resolution=resolution,cutoff=cutoff,points=[],all_timesteps=True)
        return super().solve(init, T, N, reps)

    def phi(self, pos, it):
        dir = np.array(self.importance[it][1](pos[0],pos[1]))
        fac = 1/(self.sigma*2*np.pi)/(1 + np.abs(self.importance[it][0](pos[0],pos[1])))
        return fac*dir 

class PDE_importance_sampler_sq(SDE):
    def solve(self, init, T=10, N=1000, reps=10000, resolution=100, cutoff=20.0):
        pde_solver = PDE(self.sigma,self.R,self.Q)
        self.importance = pde_solver.solve(T=T,N=N,resolution=resolution,cutoff=cutoff,points=[],all_timesteps=True)
        return super().solve(init, T, N, reps)

    def phi(self, pos, it):
        dir = np.array(self.importance[it][1](pos[0],pos[1]))
        fac = 1/(self.sigma*2*np.pi)
        return fac*dir*np.abs(self.importance[it][0](pos[0],pos[1]))

class Distance_sampler(SDE):
    def phi(self, pos, it):
        dir = pos/np.linalg.norm(pos)**2
        fac = 1/(self.sigma)*self.Q/(2*np.pi)*np.tanh(np.linalg.norm(pos)-self.R)*self.sigma**2/2
        return fac*dir