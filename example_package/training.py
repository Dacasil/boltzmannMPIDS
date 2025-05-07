import torch
from torch import optim
import matplotlib as plt
from tqdm import tqdm


def TrainBatch(
    model,
    data,
    epochs,
    learningRate,
    noise_levels,
    steps_statistics,
    annealing_scheme: float | torch.Tensor,
    training_params=None,
    n_steps=None,
    bar=True,
):
    """
    annealing_scheme: temperature, or annealing scheme
    n_steps: number of steps to equilibrium, alternative to specifying a scheme
    """
    optimizer = optim.SGD(model.parameters(), lr=learningRate)
    iterator = range(epochs)
    if bar:
        iterator = tqdm(iterator)
    for iep in iterator:
        model.training_step(
            optimizer,
            data,
            noise_levels,
            steps_statistics,
            annealing_scheme,
            n_steps,
            **training_params,
        )
