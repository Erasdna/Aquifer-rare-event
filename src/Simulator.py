class Simulator:
    """
        Baseline simulator engine
    """
    def __init__(self,sigma : float =0.5, R : float =1.0,Q : float =-10.0) -> None:
        """Initialize a generic simulation

        Args:
            sigma (float, optional): Diffusion parameter. Defaults to 0.5.
            R (float, optional): Well radius. Defaults to 1.0.
            Q (float, optional): Well perturbation parameter. Defaults to -10.0.
        """
        self.sigma=sigma
        self.R = R 
        self.Q = Q 
    
    def solve(self,T=10.0,N=1000):
        pass 

