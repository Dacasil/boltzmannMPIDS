import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np

class BoltzmannMachine(nn.Module):
    def __init__(self, nv, nh, mode='bipolar'):
        super(BoltzmannMachine, self).__init__()
        self.nv = nv  # Number of visible units
        self.nh = nh  # Number of hidden units
        self.n = nv + nh  # Total number of units
        self.w = nn.Parameter(torch.zeros(self.n, self.n))  # Full weight matrix
        self.inactiveState = -1 if mode == 'bipolar' else 0  # Inactive state
    
        # Create a connectivity mask
        self.set_backhook(self.CreateDecoderAdjacencyMat(self.nv//2,self.nh))

    def set_backhook(self,connectivity):
        def get_grad_filter(fltr):
            # used for fixing the weights that are not connected (always 0)
            def backhook(grad):
                grad = grad * fltr.to(grad.device)
                return grad
            return backhook
        self.w.register_hook(get_grad_filter(connectivity))

    def CreateDecoderAdjacencyMat(self, n, m):
        # Initialize adjacency matrix
        adj = np.zeros((2*n + m, 2*n + m))
    
        # Connections for V1 (first n units)
        for i in range(n):
            adj[i, :] = 1  # Connect to all units
            adj[i, i] = 0  # No self-connection
            adj[i, n:n + n] = 0  # No connections to V2
    
        # Connections for V2 (next n units)
        for i in range(n, 2 * n):
            adj[i, :] = 1  # Connect to all units
            adj[i, i] = 0  # No self-connection
            adj[i, 0:n] = 0  # No connections to V1
    
        # Connections for hidden units (last m units)
        for i in range(2 * n, 2 * n + m):
            adj[i, :] = 1  # Connect to all units
            adj[i, 2 * n:2 * n + m] = 0  # No connections among hidden units
    
        return torch.tensor(adj, dtype=torch.float32)


    def forward(self,initial_state, clamping_degree, T):
        perm = torch.randperm(self.n)
        for j in perm:
            if clamping_degree[j] == 0:
                sum_input = torch.dot(self.w[j], initial_state)
                p = 1 / (1 + torch.exp(-sum_input / T))
                if torch.rand(1) <= p:
                    initial_state[j] = 1
                else:
                    initial_state[j] = self.inactiveState

    
    def AddNoise(self, v, flip_probabilities):
        """
        flip_probabilities: (p_flip_zeros,p_flip_ones)
        """
        v_dash = torch.clone(v)
        for i in range(self.nv):
                if v_dash[i] == 1:
                    if torch.rand(1) <= flip_probabilities[1]:
                        v_dash[i] = self.inactiveState
                else:
                    if torch.rand(1) <= flip_probabilities[0]:
                        v_dash[i] = 1
        return v_dash

    def GoToEquilibriumState(self, initial_state:torch.Tensor, clamping_degree:torch.Tensor,
                              annealing_scheme:torch.Tensor|float,n_steps:int=None):
        """
        initial_state: concatted vector with shape (n_visible+n_hidden)
        v: state vector (not batched)
        clamping_degree: vector with degree of clamping, interpolates between initial_state and a
        T: temperature
        """
        if n_steps is not None and isinstance(annealing_scheme,(float,int)):
            annealing_scheme = annealing_scheme * torch.ones(n_steps)
        # Create the initial state of the visible and hidden units
        vhInit = torch.cat((initial_state, torch.zeros(self.nh))) 
        
        # Generate a random state for all units
        vhRandom = torch.randint(2, size=(self.n,)) * 2 - 1 if self.inactiveState == -1 else torch.randint(2, size=(self.n,))

        # Combine the clamped units (fixed to their initial values) and the free units (randomly initialized)
        total_state = clamping_degree * vhInit + (1 - clamping_degree) * vhRandom

        for temperature in annealing_scheme:  # Do one forward for each specified temperature
            self.forward(total_state,clamping_degree,T=temperature)

        return total_state[:self.nv], total_state[self.nv:] # Visible + Hidden
    
    def training_step(self,optimizer,data,noise_levels,steps_statistics,annealing_scheme,n_steps):
        nData = len(data)
        optimizer.zero_grad()
        pClampedAvg = torch.zeros(self.n, self.n)
        pFreeAvg = torch.zeros(self.n, self.n)

        for i in range(nData):
            NoisyV = self.AddNoise(data[i],noise_levels)
            vNoisy = torch.clone(NoisyV)

            clampedUnits = torch.cat((torch.ones(self.nv), torch.zeros(self.nh)))

            final_T = annealing_scheme[-1]
            vClamped, hClamped = self.GoToEquilibriumState(vNoisy, clampedUnits,
                                                            annealing_scheme,n_steps)
            pClampedAvg += self.CollectStatistics(vClamped, hClamped, clampedUnits, steps_statistics, final_T)

            vFree, hFree = self.GoToEquilibriumState(torch.zeros(self.nv), torch.zeros(self.n),
                                                      annealing_scheme, n_steps)
            final_T = annealing_scheme[-1]
            pFreeAvg += self.CollectStatistics(vFree, hFree, torch.zeros(self.n), steps_statistics, final_T)

        pClampedAvg /= nData
        pFreeAvg /= nData
        s = pClampedAvg - pFreeAvg

        self.w.grad = -s  # Gradient descent
        optimizer.step()
    
    def CollectStatistics(self, v, h, clamped, timeUnits, T):
            vh = torch.cat((v, h))
            stats = torch.zeros(self.n, self.n)
    
            for _ in range(timeUnits):
                self.forward(vh,clamped, T)
                stats += torch.outer(vh, vh)
    
            return stats / timeUnits