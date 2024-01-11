import numpy as np 

class Simulator:
    def __init__(self,sigma=0.5, R=1.0,Q=-10.0) -> None:
        self.sigma=sigma
        self.R = R 
        self.Q = Q 
    
    def solve(self,T=10.0,N=1000):
        pass 

