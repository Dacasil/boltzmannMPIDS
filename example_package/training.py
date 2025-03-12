import torch
from torch import optim
import matplotlib as plt

def TrainBatch(model, data, epochs, learningRate,noise_levels,steps_statistics,train_params:dict,
               annealing_scheme:float|torch.Tensor,n_steps=None):
        """
        annealing_scheme: temperature, or annealing scheme
        n_steps: number of steps to equilibrium, alternative to specifying a scheme 
        """
        optimizer = optim.SGD(model.parameters(), lr=learningRate)
        for iep in range(epochs):
            model.training_step(optimizer,data,noise_levels,steps_statistics,annealing_scheme,n_steps,
                                **train_params)
            
            

