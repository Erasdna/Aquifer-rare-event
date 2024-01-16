import numpy as np
from tqdm import tqdm

from Simulator import Simulator
from PDE_simulator import PDE


class SDE(Simulator):
    """
    Solver engine using SDE
    """

    def __init__(
        self, sigma: float = 0.5, R: float = 1.0, Q: float = -10.0, seed: int = 55
    ) -> None:
        """Initialize SDE MC simulator with seed

        Args:
            sigma (float, optional): Diffusion parameter. Defaults to 0.5.
            R (float, optional): Well radius. Defaults to 1.0.
            Q (float, optional): Well perturbation parameter. Defaults to -10.0.
            seed (int, optional): Seed. Defaults to 55.
        """
        super().__init__(sigma, R, Q)
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

    def solve(
        self,
        init: np.ndarray,
        T: float = 10.0,
        N: int = 1000,
        reps: int = 10000,
        save_pos: bool = False,
    ) -> list:
        """Solve the problem using a SDE MC estimator

        Args:
            init (np.ndarray): Starting position
            T (float, optional): Final time. Defaults to 10.0.
            N (int, optional): Number of time-steps. Defaults to 1000.
            reps (int, optional): Number of MC iterations. Defaults to 10000.
            save_pos (bool, optional): If True saves the position at each time-step. Defaults to False.

        Returns:
            list: Stopping state and if save_pos is True, the position at each time-step
        """
        stop = np.zeros(reps)
        if save_pos:
            histories = []
        for it in tqdm(range(reps)):
            pos = init.copy()

            if save_pos:
                stop[it], _, _, tmp = self.sim(pos, T, N, save_pos=save_pos)
                histories.append(tmp)
            else:
                stop[it], _, _ = self.sim(pos, T, N)
        if save_pos:
            return stop, histories
        else:
            return stop

    def flow(self, pos: np.ndarray) -> np.ndarray:
        """Flow field

        Args:
            pos (np.ndarray): position

        Returns:
            np.ndarry: Vector value of u
        """

        steady = np.array([1.0, 0.0])
        perturbation = pos / np.linalg.norm(pos) ** 2
        return steady + (self.Q / (2 * np.pi)) * perturbation

    def path(self, pos: np.ndarray, it: int, dt: float) -> [np.ndarray, float]:
        """Samples a Gaussian path

        Args:
            pos (np.ndarray): position
            it (int): Iteration
            dt (float): Time-step

        Returns:
            (np.ndarray,float): Step to be taken and the log of the weight parameter
        """

        shift = self.phi(pos, it)
        step = self.rng.normal(loc=shift * dt, scale=np.sqrt(dt), size=(2,))
        log_weight = 0.5 * dt * np.linalg.norm(shift) ** 2 - shift.T @ step
        return step, log_weight

    def phi(self, pos: np.ndarray, it: int) -> np.ndarray:
        """The importance sampling flow-modification term. Used in the variance reducion methods

        Args:
            pos (np.ndarray): position
            it (int): iteration

        Returns:
            np.ndarray: value of phi
        """
        return np.zeros(2)

    def sim(
        self,
        pos: np.ndarray,
        T: float,
        N: int,
        t0: float = 0.0,
        condition: float = None,
        save_pos: bool = False,
    ) -> [float,float,np.ndarray]:
        """Simulates a realization of the SDE

        Args:
            pos (np.ndarray): Starting position
            T (float): Final time
            N (int): Number of time-steps
            t0 (float, optional): Starting time. Defaults to 0.0.
            condition (float, optional): Stopping condition other than reaching the inner well. Defaults to None.
            save_pos (bool, optional): If True all positions are saved. Defaults to False.

        Returns:
            float,np.ndarray,float, (list, optional): weight parameter, final time, final position and optionally all positions
        """
        if save_pos:
            history = []

        if condition is not None:
            r = condition
        else:
            r = self.R

        dt = T / N
        n = int(N - t0 / dt)

        logw = 0 #Logarithm of weight parameter
        for i, t in enumerate(np.linspace(t0, T, n)):
            if save_pos:
                history.append(pos.copy())
            if np.linalg.norm(pos) <= r or np.linalg.norm(pos) <= self.R:
                if save_pos:
                    return 1.0 * np.exp(logw), t, pos, history
                else:
                    return 1.0 * np.exp(logw), t, pos

            xi, tmp = self.path(pos, N - i - 1, dt)
            logw += tmp
            pos += self.flow(pos) * dt + self.sigma * xi
        if save_pos:
            return 0.0, T, pos, history
        else:
            return 0.0, T, pos


class SDE_path(SDE):
    """
        Utility class for generating a MC SDE simulation from a pre-defined path
    """
    def solve(self, init, preloaded_path, T=10, N=1000, reps=10000):
        self.preloaded_path = preloaded_path
        stop = np.zeros(reps)
        for it in tqdm(range(reps)):
            pos = init.copy()
            self.current_path = self.preloaded_path[it, :, :]
            stop[it], _, _ = self.sim(pos, T, N)

        return np.mean(stop), np.std(stop)

    def path(self, pos, it, dt):
        return self.current_path[it, :], 0.0
