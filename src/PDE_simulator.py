import numpy as np 
from tqdm import tqdm
from Simulator import Simulator

from fenics import *
from dolfin import *
from mshr import *

class PDE(Simulator):    
    def _make_mesh(self,cutoff,resolution):
        outer = Circle(Point(0,0),cutoff)
        domain = outer
        return generate_mesh(domain,resolution)
    
    def solve(self,T=10, N=1000, resolution=100, cutoff=20.0, points=np.array([[1.2,1.1],[2.5,2.5],[3.0,4.0]]), verbose=True, all_timesteps=False):
        dt = T/N

        #Build mesh and function spaces
        mesh = self._make_mesh(cutoff,resolution)
        V = FunctionSpace(mesh,"P",1)
        gradV = VectorFunctionSpace(mesh,"P",1)
        
        #Boundary condition
        u_boundary = Expression('x[0]*x[0] + x[1]*x[1]<=R*R ? 1.0 : 0.0',R=self.R,degree=2)
        def boundary(x,on_boundary):
            return np.linalg.norm(x)<=self.R or on_boundary
    
        bc = DirichletBC(V,u_boundary,boundary)

        #Initial condition
        IV = interpolate(u_boundary,V)

        u = TrialFunction(V)
        v = TestFunction(V)

        flow = Expression(("1 + (Q/(2*pi))*x[0]/(x[0]*x[0] + x[1]*x[1])", "(Q/(2*pi))*x[1]/(x[0]*x[0] + x[1]*x[1])"),Q=self.Q,pi=np.pi,degree=2)

        #Set up weak formulation
        F = u*v*dx - IV*v*dx - dt*dot(flow,grad(u))*v*dx + dt*(self.sigma**2/2)*dot(grad(u),grad(v))*dx
        a, L = lhs(F),rhs(F)

        if all_timesteps:
            history=[]

        u=Function(V)
        t=T

        for t in np.linspace(T - dt,0,N):
            solve(a==L,u,bc,solver_parameters={'linear_solver': 'gmres',
                                               'preconditioner': 'ilu'})
            if all_timesteps:
                history.append([u.copy(),project(grad(u),gradV)])
            IV.assign(u) #update old state
            if verbose:
                print("Time: ", t)
                for p in points:
                    print(p,": ", u(p[0],p[1]))

        if all_timesteps:
            return history
        else:
            return u