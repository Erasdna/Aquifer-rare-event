import numpy as np
from tqdm import tqdm
from Simulator import Simulator

from fenics import *
from dolfin import *
from mshr import *


class PDE(Simulator):
    """
        Solver engine for the Feynman-Kac formula 
    """
    def _make_mesh(self, cutoff: float, resolution: int):
        outer = Circle(Point(0, 0), cutoff)
        domain = outer
        return generate_mesh(domain, resolution)

    def solve(
        self,
        T : float =10.0,
        N : int =1000,
        resolution : int =100,
        cutoff : float =20.0,
        points=np.array([[1.2, 1.1], [2.5, 2.5], [3.0, 4.0]]),
        verbose : bool =True,
        all_timesteps : bool =False,
        return_grad : bool =False,
    ) -> list :
        """Solve the Feynman-Kac formula using finite elements and backwards Euler time-stepping

        Args:
            T (float, optional): Final time. Defaults to 10.0.
            N (int, optional): Number of timesteps. Defaults to 1000.
            resolution (int, optional): Mesh resolution. Defaults to 100.
            cutoff (float, optional): Domain cutoff. Defaults to 20.0.
            points (np.ndarray, optional): Points where the probability is displayed. Defaults to np.array([[1.2, 1.1], [2.5, 2.5], [3.0, 4.0]]).
            verbose (bool, optional): Choose to display simulation at each time-step. Defaults to True.
            all_timesteps (bool, optional): If True returns function u in all timesteps. Defaults to False.
            return_grad (bool, optional): Return gradient of the function. Defaults to False.

        Returns:
            list: list of solutions
        """
        dt = T / N

        # Build mesh and function spaces
        mesh = self._make_mesh(cutoff, resolution)
        V = FunctionSpace(mesh, "P", 1)
        gradV = VectorFunctionSpace(mesh, "P", 1)

        # Boundary condition
        u_boundary = Expression(
            "x[0]*x[0] + x[1]*x[1]<=R*R ? 1.0 : 0.0", R=self.R, degree=2
        )

        def boundary(x, on_boundary):
            return np.linalg.norm(x) <= self.R or on_boundary

        bc = DirichletBC(V, u_boundary, boundary)

        # Initial condition
        IV = interpolate(u_boundary, V)

        u = TrialFunction(V)
        v = TestFunction(V)

        #Fenics C expression for the flow-field
        flow = Expression(
            (
                "1 + (Q/(2*pi))*x[0]/(x[0]*x[0] + x[1]*x[1])",
                "(Q/(2*pi))*x[1]/(x[0]*x[0] + x[1]*x[1])",
            ),
            Q=self.Q,
            pi=np.pi,
            degree=2,
        )

        # Set up weak formulation
        F = (
            u * v * dx
            - IV * v * dx
            - dt * dot(flow, grad(u)) * v * dx
            + dt * (self.sigma**2 / 2) * dot(grad(u), grad(v)) * dx
        )
        a, L = lhs(F), rhs(F)

        if all_timesteps:
            history = []

        u = Function(V)
        t = T

        #Time-iteration
        for t in np.linspace(T - dt, 0, N):
            solve(
                a == L,
                u,
                bc,
                solver_parameters={"linear_solver": "gmres", "preconditioner": "ilu"},
            )
            if all_timesteps:
                if not return_grad:
                    history.append([u.copy(), 0])  # project(grad(u),gradV)])
                else:
                    history.append([u.copy(), project(grad(u), gradV)])
            IV.assign(u)  # update old state
            if verbose:
                print("Time: ", t)
                for p in points:
                    print(p, ": ", u(p[0], p[1]))

        if all_timesteps:
            return history
        else:
            return u
