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
            
            if iep in plot_intervals:
                PlotWeights(model,iep)

def PlotWeights_simple(weights, epoch):
    plt.figure(figsize=(8, 6))
    plt.imshow(weights, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(f'Weight Matrix at Epoch {epoch}')
    plt.xlabel('Units')
    plt.ylabel('Units')
    plt.show()

def PlotWeights(model, epoch):
    # Get the weight matrix
    weight_matrix = model.w.detach().cpu().numpy()
    
    # Define the structure of the network
    nv1 = model.nv // 2  # Number of visible units
    nv2 = model.nv // 2  # Number of visible units
    nh = model.nh        # Number of hidden units
    
    # Create a figure for the entire network
    fig, axes = plt.subplots(nv1 + nh + nv2, 1, figsize=(10, 2 * (nv1 + nh + nv2)))
    
    # Define labels for the units
    unit_labels = [f'V1_{i+1}' for i in range(nv1)] + \
                    [f'H_{i+1}' for i in range(nh)] + \
                    [f'V2_{i+1}' for i in range(nv2)]
    
    # Iterate over each unit and plot its connections
    for i in range(model.n):
        # Create a subplot for the current unit
        ax = axes[i]
        
        # Extract the weights for the current unit (1D array of size 10)
        unit_weights = weight_matrix[i, :]
        
        # Reshape the weights into a 2D grid for visualization
        unit_weights_grid = unit_weights.reshape(1, -1)  # Reshape to (1, 10)
        
        # Plot the weights as a heatmap
        im = ax.imshow(unit_weights_grid, cmap='viridis', vmin=-1, vmax=1, aspect='auto')
        
        # Add labels to the x-axis
        ax.set_xticks(range(model.n))
        ax.set_xticklabels(unit_labels, rotation=90)
        
        # Add a title to the subplot
        ax.set_title(f'Unit {i+1} ({unit_labels[i]}) Connections')
        
        # Hide y-axis ticks
        ax.set_yticks([])
        
        # Add a colorbar for each subplot
        plt.colorbar(im, ax=ax)
        
        # Add background shading to distinguish visible and hidden units
        for j in range(model.n):
            if j < nv1:
                ax.axvspan(j - 0.5, j + 0.5, color='lightblue', alpha=0.3)  # V1 units
            elif j < nv1 + nh:
                ax.axvspan(j - 0.5, j + 0.5, color='lightgreen', alpha=0.3)  # Hidden units
            else:
                ax.axvspan(j - 0.5, j + 0.5, color='lightcoral', alpha=0.3)  # V2 units
    
    # Adjust layout and display the figure
    plt.tight_layout()
    plt.suptitle(f'Weight Visualization at Epoch {epoch}', y=1.02)
    plt.show()