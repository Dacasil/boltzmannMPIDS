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


    def forward(self, v, clamped, T):
        return self.GoToEquilibriumState(v, clamped, T)

    
    def AddNoise(self, v, flip_probabilities):
        """
        flip_probabilities: (p_flip_to_-1,p_flip_to_one)
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

    def GoToEquilibriumState(self, v:torch.Tensor, clamping_degree:torch.Tensor, T:float):
        """
        v: state vector (not batched)
        clamping_degree: vector with degree of clamping, interpolates
        T: temperature
        """
        # Create the initial state of the visible and hidden units
        vhInit = torch.cat((v, torch.zeros(self.nh))) 
        
        # Generate a random state for all units
        vhRandom = torch.randint(2, size=(self.n,)) * 2 - 1 if self.inactiveState == -1 else torch.randint(2, size=(self.n,))

        # Combine the clamped units (fixed to their initial values) and the free units (randomly initialized)
        vh = clamping_degree * vhInit + (1 - clamping_degree) * vhRandom

        for _ in range(100):  # Fixed number of steps
            perm = torch.randperm(self.n) # Generates a random order for updating the units
            for j in perm:
                if clamping_degree[j] == 0:
                    sum_input = torch.dot(self.w[j], vh)
                    p = 1 / (1 + torch.exp(-sum_input / T))
                    if torch.rand(1) <= p:
                        vh[j] = 1
                    else:
                        vh[j] = self.inactiveState

        return vh[:self.nv], vh[self.nv:] # Visible + Hidden
    
    def training_step(self,optimizer,data,noise_levels,T):
        nData = len(data)
        optimizer.zero_grad()
        pClampedAvg = torch.zeros(self.n, self.n)
        pFreeAvg = torch.zeros(self.n, self.n)

        for i in range(nData):
            NoisyV = self.AddNoise(data[i],noise_levels)
            vNoisy = torch.clone(NoisyV)

            clampedUnits = torch.cat((torch.ones(self.nv), torch.zeros(self.nh)))
            vClamped, hClamped = self.GoToEquilibriumState(vNoisy, clampedUnits, T)
            pClampedAvg += self.CollectStatistics(vClamped, hClamped, clampedUnits, 10, T)

            vFree, hFree = self.GoToEquilibriumState(torch.zeros(self.nv), torch.zeros(self.n), T)
            pFreeAvg += self.CollectStatistics(vFree, hFree, torch.zeros(self.n), 10, T)

        pClampedAvg /= nData
        pFreeAvg /= nData
        s = pClampedAvg - pFreeAvg

        self.w.grad = -s  # Gradient descent
        optimizer.step()
    
    def CollectStatistics(self, v, h, clamped, timeUnits, T):
            vh = torch.cat((v, h))
            stats = torch.zeros(self.n, self.n)
    
            for _ in range(timeUnits):
                perm = torch.randperm(self.n)
                for j in perm:
                    if clamped[j] == 0:
                        sum_input = torch.dot(self.w[j], vh)
                        p = 1 / (1 + torch.exp(-sum_input / T))
                        if torch.rand(1) <= p:
                            vh[j] = 1
                        else:
                            vh[j] = self.inactiveState
                stats += torch.outer(vh, vh)
    
            return stats / timeUnits