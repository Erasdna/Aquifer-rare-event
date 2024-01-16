import numpy as np
from tqdm import tqdm

from SDE_simulator import SDE


class SDE_splitting(SDE):
    def solve(
        self,
        init: np.ndarray,
        num_levels : int,
        split : str ="linear",
        T : float =10.0,
        N : int =1000,
        reps : int =10000,
        scale : str ="lin",
        verbose : bool =False,
    ):
        """Sovler engine using the splitting method for the MC SDE simulation

        Args:
            init (np.ndarray): Starting position
            num_levels (int): Number of cocentric levels
            split (str, optional): Level spacing method. Defaults to "linear".
            T (float, optional): Final time. Defaults to 10.0.
            N (int, optional): Number of time-steps. Defaults to 1000.
            reps (int, optional): Number of MC iterations. Defaults to 10000.
            scale (str, optional): Decay of MC iterations. Defaults to "lin".
            verbose (bool, optional): If True information on the simulation is displayed. Defaults to False.
        """
        assert num_levels > 1

        # Impose descending
        if split == "linear":
            levels = linear_levels(num_levels, 1.0, np.linalg.norm(init))
        elif split == "log":
            levels = log_levels(num_levels, 1.0, np.linalg.norm(init))
            
        levels = levels[np.argsort(-levels)]
        level_prob = np.zeros_like(levels)
        #cumulative probability
        cumulative = 1.0
        it_rep = reps

        init_list = [init]
        t0 = [0.0]
        #Iterate levels
        for nl, l in enumerate(levels):
            if verbose:
                print("Level: ", l)
            new_init = []
            new_t0 = []

            #If scale is not recognized the nb of iterations stays constant
            if scale == "lin":
                it_rep = int(reps / len(init_list)) + 2
            elif scale == "sq":
                it_rep = int(reps / len(init_list) ** 2) + 2
            elif scale == "exp":
                it_rep = int(reps / np.exp(len(init_list))) + 2

            if verbose:
                print("Reps: ", it_rep)

            stop = np.zeros((it_rep, len(init_list)))
            for i, init in enumerate(init_list):
                if verbose:
                    print(
                        "Starting point: ",
                        i + 1,
                        "/",
                        len(init_list),
                        "point: ",
                        init,
                        "time: ",
                        t0[i],
                    )
                for it in range(it_rep):
                    start = init.copy()
                    stop[it, i], tmp_t, tmp_pos = self.sim(
                        start, T, N, t0=t0[i], condition=l
                    )
                    if tmp_t < T:
                        new_init.append(tmp_pos)
                        new_t0.append(tmp_t)
            level_prob[nl] = np.mean(stop)
            if verbose:
                print("Probability at level ", l, " ", level_prob[nl])
            cumulative *= level_prob[nl]

            init_list = new_init.copy()
            t0 = new_t0.copy()

            if len(init_list) < 1:
                return 0.0, level_prob

        return cumulative, level_prob


def linear_levels(n, start, stop):
    return np.linspace(start, stop, n + 1)[:-1]


def log_levels(n, start, stop):
    return np.exp(np.linspace(np.log(start), np.log(stop), n + 1))[:-1]
