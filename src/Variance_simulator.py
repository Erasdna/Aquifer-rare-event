import numpy as np 

from PDE_simulator import PDE
from SDE_simulator import SDE

class PDE_importance(SDE):
    def solve(self, init, T=10, N=1000, reps=10000, resolution=100, cutoff=20.0,pre_run=None,save_pos=False):
        if pre_run is not None:
            self.importance = pre_run
        else:
            pde_solver = PDE(self.sigma,self.R,self.Q)
            self.importance = pde_solver.solve(T=T,N=N,resolution=resolution,cutoff=cutoff,points=[],all_timesteps=True, return_grad=False)

        return super().solve(init, T, N, reps,save_pos=save_pos)
    
class dynamic_Q(PDE_importance):
    def phi(self, pos, it):
        perturbation = pos/np.linalg.norm(pos)**2
        fac = (self.Q/(2*np.pi))*(2/(1 + self.importance[it][0](pos[0],pos[1])) - 1)
        return (1/self.sigma)*fac*perturbation

class dynamic_steady(PDE_importance):
    def phi(self, pos, it):
        dir = np.array([self.importance[it][0](pos[0],pos[1])-1.0,0.0])
        return (1/self.sigma)*dir

class dynamic_combo(PDE_importance):
    def phi(self, pos, it):
        perturbation = pos/np.linalg.norm(pos)**2
        fac = (self.Q/(2*np.pi))*(2/(1 + self.importance[it][0](pos[0],pos[1])) - 1)
        dir = np.array([self.importance[it][0](pos[0],pos[1])-1.0,0.0])
        return (1/self.sigma)*(dir + fac*perturbation)
    


# class phi_log(PDE_importance):
#     def phi(self, pos, it):
#         dir = np.array(self.importance[it][1](pos[0],pos[1]))
#         fac = 1/(self.sigma*2*np.pi)/(1 + np.abs(self.importance[it][0](pos[0],pos[1])))
#         return fac*dir*self.alpha 

# class phi_sq(PDE_importance):
#     def phi(self, pos, it):
#         dir = np.array(self.importance[it][1](pos[0],pos[1]))
#         fac = 1/(self.sigma*2*np.pi)
#         return fac*dir*np.abs(self.importance[it][0](pos[0],pos[1]))*self.alpha

# class phi(PDE_importance):
#     def phi(self, pos, it):
#         dir = np.array(self.importance[it][1](pos[0],pos[1]))
#         fac = 1/(self.sigma*2*np.pi)
#         return fac*dir*self.alpha

# class full(PDE_importance):
#     def phi(self, pos, it):
#         dir = np.array(self.importance[it][1](pos[0],pos[1]))
#         fac = 1/(self.sigma*2*np.pi)
#         return fac*dir*self.alpha - (1/self.sigma)*self.flow(pos)
    
# class PDE_importance_Q(SDE):
#     def solve(self, init, T=10, N=1000, reps=10000, resolution=100, cutoff=20.0,alpha=1.0,pre_run=None):
#         self.alpha=alpha

#         if pre_run is not None:
#             self.importance = pre_run
#         else:
#             pde_solver = PDE(self.sigma,self.R,self.Q)
#             self.importance = pde_solver.solve(T=T,N=N,resolution=resolution,cutoff=cutoff,points=[],all_timesteps=True, return_grad=False)

#         return super().solve(init, T, N, reps)

    

