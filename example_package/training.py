import torch
from torch import optim
import matplotlib as plt
import example_package

def TrainBatch(model:example_package.BoltzmannMachine, data, epochs, learningRate,noise_levels,T:float|torch.Tensor, plot_intervals=[]):
        """
        T: temperature, or annealing scheme
        """
        optimizer = optim.SGD(model.parameters(), lr=learningRate)
    
        for iep in range(epochs):
            model.training_step(optimizer,data,noise_levels,T)
            
            

