import numpy as np

from PDE_simulator import PDE
from SDE_simulator import SDE


class PDE_importance(SDE):
    def solve(
        self,
        init: np.ndarray,
        T: float = 10.0,
        N: int = 1000,
        reps: int = 10000,
        resolution: int = 100,
        cutoff: float = 20.0,
        pre_run: any = None,
        save_pos: bool = False,
    ):
        """Solve the SDE using MC importance sampling based on results using Feynman-Kac

        Args:
            init (np.ndarray): Starting position
            T (float, optional): Final time. Defaults to 10.0.
            N (int, optional): Number of time-steps. Defaults to 1000.
            reps (int, optional): Number of MC iterations. Defaults to 10000.
            resolution (int, optional): Mesh resolution. Defaults to 100.
            cutoff (float, optional): Domain cutoff. Defaults to 20.0.
            pre_run (any, optional): Pre-run Feynman-Kac function. Defaults to None.
            save_pos (bool, optional): If True all time-step positions are saved. Defaults to False.

        Returns:
           See parent class
        """
        if pre_run is not None:
            self.importance = pre_run
        else:
            pde_solver = PDE(self.sigma, self.R, self.Q)
            self.importance = pde_solver.solve(
                T=T,
                N=N,
                resolution=resolution,
                cutoff=cutoff,
                points=[],
                all_timesteps=True,
                return_grad=False,
            )

        return super().solve(init, T, N, reps, save_pos=save_pos)


class dynamic_Q(PDE_importance):
    """
    Importance sampling modifying the value of the well parameter Q
    """

    def phi(self, pos: np.ndarray, it: int):
        perturbation = pos / np.linalg.norm(pos) ** 2
        fac = (self.Q / (2 * np.pi)) * (
            2 / (1 + self.importance[it][0](pos[0], pos[1])) - 1
        )
        return (1 / self.sigma) * fac * perturbation


class dynamic_steady(PDE_importance):
    """
    Importance sampling modifying the steady flow
    """

    def phi(self, pos: np.ndarray, it: int):
        dir = np.array([self.importance[it][0](pos[0], pos[1]) - 1.0, 0.0])
        return (1 / self.sigma) * dir


class dynamic_combo(PDE_importance):
    """
    Importance sampling modifying the steady flow and the well parameter Q
    """

    def phi(self, pos: np.ndarray, it: int):
        perturbation = pos / np.linalg.norm(pos) ** 2
        fac = (self.Q / (2 * np.pi)) * (
            2 / (1 + self.importance[it][0](pos[0], pos[1])) - 1
        )
        dir = np.array([self.importance[it][0](pos[0], pos[1]) - 1.0, 0.0])
        return (1 / self.sigma) * (dir + fac * perturbation)
