from fenics import *
from dolfin import *
from mshr import *
import numpy as np

def make_mesh(outer_radius,mesh_size):
    outer = Circle(Point(0,0),outer_radius)
    domain = outer
    return generate_mesh(domain,mesh_size)

def PDE_solve(N,sigma=0.25,R=1.0,T=10.0, Q=-7.0, cutoff = 20.0,dof=200, points=[[1.2,1.1],[2.5,2.5],[3.0,4.0]], verbose=True):
    dt=T/N
    
    mesh = make_mesh(cutoff,dof)

    V = FunctionSpace(mesh,"P",1)
    #Should only be called on the boundary
    u_boundary = Expression('x[0]*x[0] + x[1]*x[1]<=R*R ? 1.0 : 0.0',R=R,degree=2)

    def boundary(x,on_boundary):
        return np.linalg.norm(x)<=R or on_boundary
    
    bc = DirichletBC(V,u_boundary,boundary)

    IV = interpolate(u_boundary,V)

    u = TrialFunction(V)
    v = TestFunction(V)

    flow = Expression(("1 + (Q/(2*pi))*x[0]/(x[0]*x[0] + x[1]*x[1])", "(Q/(2*pi))*x[1]/(x[0]*x[0] + x[1]*x[1])"),Q=Q,pi=np.pi,degree=2)
    #We solve from terminal condition => tau = T - t
    #The convective term goes in the opposite direction due to time inversion
    F = u*v*dx - IV*v*dx - dt*dot(flow,grad(u))*v*dx + dt*(sigma**2/2)*dot(grad(u),grad(v))*dx
    a, L = lhs(F),rhs(F)

    u=Function(V)
    t=T

    while t>dt:
        print(t)
        solve(a==L,u,bc)
        IV.assign(u) #update old state
        t-=dt
        print(len(u.vector()))
        if verbose:
            for p in points:
                print(p,": ", u(p[0],p[1]))
    return u