import numpy as np
from tqdm import tqdm

from SDE_simulator import SDE

class SDE_splitting(SDE):
    def solve(self, init, levels, T=10, N=1000, reps=10000, scale = 'lin', verbose=False):
        assert(len(levels)>1)
        #Impose descending
        levels = levels[np.argsort(-levels)]
        level_prob = np.zeros_like(levels)
        cumulative = 1.0
        it_rep = reps

        init_list = [init]
        t0 = [0.0]
        for nl,l in enumerate(levels):
            if verbose:
                print("Level: ", l)
            new_init=[]
            new_t0 = []

            if scale=='lin':
                it_rep = int(reps/len(init_list))+2
            elif scale=='sq':
                it_rep = int(reps/len(init_list)**2)+2
            elif scale=='exp':
                it_rep = int(reps/np.exp(len(init_list)))+2

            if verbose:
                print("Reps: ", it_rep)
                
            stop = np.zeros((it_rep,len(init_list)))
            for i, init in enumerate(init_list):
                if verbose:
                    print("Starting point: ", i+1, "/", len(init_list), "point: ", init, "time: ", t0[i])
                for it in range(it_rep):
                    start = init.copy()
                    stop[it,i],tmp_t,tmp_pos = self.sim(start,T,N,t0 = t0[i], condition=l)
                    if tmp_t<T:
                        new_init.append(tmp_pos)
                        new_t0.append(tmp_t)
            level_prob[nl] = np.mean(stop)
            if verbose:
                print("Probability at level ", l, " ", level_prob[nl])
            cumulative*=level_prob[nl]

            init_list = new_init.copy()
            t0 = new_t0.copy()

            if len(init_list)<1:
                return 0.0, level_prob

        return cumulative,level_prob
