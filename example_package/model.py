import torch
import torch.nn as nn

# import torch.optim as optim
import numpy as np


class BoltzmannMachine(nn.Module):
    def __init__(self, nv, nh, adjacency_matrix=None, bias=False):
        super(BoltzmannMachine, self).__init__()
        self.nv = nv  # Number of visible units
        self.nh = nh  # Number of hidden units
        self.bias = bias
        self.n = nv + nh + int(bias)  # Total number of units
        self.w = nn.Parameter(torch.zeros(self.n, self.n))
        self.inactiveState = 0  # CHANGED: Fixed to binary units (0)

        # Compute the default adjacency matrix if none is provided
        if adjacency_matrix is None:
            self.adjacency_matrix = AdjMat_V1V2_inlayer(nv // 2, nh, bias)
        else:
            self.adjacency_matrix = adjacency_matrix

        # Create a connectivity mask
        self.set_backhook(self.adjacency_matrix)

    def set_backhook(self, connectivity):
        def get_grad_filter(fltr):
            # Used for fixing the weights that are not connected (always 0)
            def backhook(grad):
                grad = grad * fltr.to(grad.device)
                return grad

            return backhook

        self.w.register_hook(get_grad_filter(connectivity))

    def forward(self, initial_state, clamping_degree, T):
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

    def GoToEquilibriumState(
        self,
        initial_state: torch.Tensor,
        clamping_degree: torch.Tensor,
        annealing_scheme: torch.Tensor | float,
        n_steps: int = None,
    ):
        if n_steps is not None and isinstance(annealing_scheme, (float, int)):
            annealing_scheme = annealing_scheme * torch.ones(n_steps)

        # Initialize all units randomly (binary)
        total_state = torch.randint(2, size=(self.n,), dtype=torch.float32)

        # Ensure initial_state matches the size of total_state
        if len(initial_state) < self.n:
            # Pad initial_state with zeros for hidden units and bias
            padding_size = self.n - len(initial_state)
            initial_state = torch.cat((initial_state, torch.zeros(padding_size)))

        # Ensure clamping_degree matches the size of total_state
        if len(clamping_degree) < self.n:
            # Pad clamping_degree with zeros for hidden units and bias
            padding_size = self.n - len(clamping_degree)
            clamping_degree = torch.cat((clamping_degree, torch.zeros(padding_size)))

        # Directly set clamped units to their initial values
        total_state[clamping_degree == 1] = initial_state[clamping_degree == 1]

        # Handle bias unit if enabled
        if self.bias:
            # Ensure the bias unit is always clamped to 1
            total_state[-1] = 1  # Bias unit is always on
            clamping_degree[-1] = 1  # Clamp the bias unit

        for temperature in annealing_scheme:
            self.forward(total_state, clamping_degree, T=temperature)

        final_step = self.n - 1 if self.bias else self.n
        return (
            total_state[: self.nv],
            total_state[self.nv : final_step],
        )

    def training_step(
        self,
        optimizer,
        data,
        noise_levels,
        steps_statistics,
        annealing_scheme,
        n_steps,
        discretize_gradients=False,
        double_clamped=False,  # optional
    ):
        nData = len(data)
        optimizer.zero_grad()
        pClampedAvg = torch.zeros(self.n, self.n)
        pFreeAvg = torch.zeros(self.n, self.n)

        for i in range(nData):
            NoisyV = self.AddNoise(data[i], noise_levels)
            vNoisy = torch.clone(NoisyV)

            clampedUnits = {torch.cat((torch.ones(self.nv), torch.zeros(self.nh)))}
            if self.bias:
                clampedUnits = torch.cat(
                    (clampedUnits, torch.tensor([1.0]))
                )  # Clamp bias unit

            final_T = annealing_scheme[-1]
            n_reps = 2 if double_clamped else 1
            for _ in range(n_reps):
                vClamped, hClamped = self.GoToEquilibriumState(
                    vNoisy, clampedUnits, annealing_scheme, n_steps
                )
                pClampedAvg += self.CollectStatistics(
                    vClamped, hClamped, clampedUnits, steps_statistics, final_T
                )
            pClampedAvg = pClampedAvg / n_reps

            vFree, hFree = self.GoToEquilibriumState(
                torch.zeros(self.nv),
                torch.zeros(self.n - int(self.bias)),
                annealing_scheme,
                n_steps,
            )
            final_T = annealing_scheme[-1]
            pFreeAvg += self.CollectStatistics(
                vFree,
                hFree,
                torch.zeros(self.n - int(self.bias)),
                steps_statistics,
                final_T,
            )

        pClampedAvg /= nData
        pFreeAvg /= nData
        s = pClampedAvg - pFreeAvg
        if discretize_gradients:
            s = torch.sign(s)
        self.w.grad = -s  # Gradient descent
        optimizer.step()
        with torch.no_grad():
            self.w *= self.adjacency_matrix.to(self.w.device)

    def append_bias(self, state):
        y = torch.cat((state, torch.tensor([1])))
        return y

    def CollectStatistics(self, v, h, clamped, timeUnits, T):
        clamped = self.append_bias(clamped)
        total_state = torch.cat((v, h))
        if self.bias:
            total_state = torch.cat(
                (total_state, torch.tensor([1.0]))
            )  # Bias unit is always on
        stats = torch.zeros(self.n, self.n)

        # Ensure statistics are collected at final temperature
        for _ in range(timeUnits):
            self.forward(total_state, clamped, T)
            stats += torch.outer(total_state, total_state)

        return stats / timeUnits


def AdjMat_V1V2_inlayer(n, m, bias):
    # Initialize adjacency matrix
    total_length = 2 * n + m + int(bias)
    adj = np.zeros((total_length, total_length))

    # Connections for V1
    for i in range(n):
        adj[i, :] = 1
        adj[i, i] = 0
        adj[i, n : n + n] = 0

    # Connections for V2
    for i in range(n, 2 * n):
        adj[i, :] = 1
        adj[i, i] = 0
        adj[i, 0:n] = 0

    # Connections for hidden units
    for i in range(2 * n, 2 * n + m):
        adj[i, :] = 1
        adj[i, 2 * n : 2 * n + m] = 0

    # connect bias to all
    if bias:
        adj[total_length - 1, :] = 1
        adj[:, total_length - 1] = 1

    return torch.tensor(adj, dtype=torch.float32)


def AdjMat_V1V2_no_inlayer(n, m, bias):
    """
    n: length of V1
    """
    # Initialize adjacency matrix
    total_length = 2 * n + m + int(bias)
    adj = np.zeros((total_length, total_length))

    # Connections for V1
    for i in range(n):
        adj[i, 2 * n : 2 * n + m] = 1
        adj[i, i] = 0
        adj[i, n : 2 * n] = 0
        adj[i, 0:n] = 0

    # Connections for V2
    for i in range(n, 2 * n):
        adj[i, 2 * n : 2 * n + m] = 1
        adj[i, i] = 0
        adj[i, 0:n] = 0
        adj[i, n : 2 * n] = 0

    # Connections for hidden units
    for i in range(2 * n, 2 * n + m):
        adj[i, :] = 1
        adj[i, 2 * n : 2 * n + m] = 0

    # Connect bias to all
    if bias:
        adj[total_length - 1, :] = 1
        adj[:, total_length - 1] = 1

    return torch.tensor(adj, dtype=torch.float32)


# def main():
#     import training
#     n_visible = 4
#     n_hidden = 2
#     # Define the training set, there are n_visible possible patterns
#     v_active = torch.tensor([[1, 0, 0, 0, 1, 0, 0, 0],
#                             [0, 1, 0, 0, 0, 1, 0, 0],
#                             [0, 0, 1, 0, 0, 0, 1, 0],
#                             [0, 0, 0, 1, 0, 0, 0, 1]], dtype=torch.float32)
#     dataset = -torch.ones((n_visible,2*n_visible))+2*v_active
#     #noise_levels = [0.05,0.15]
#     epochs = 50
#     learning_rate = 2
#     noise_levels = [0.05,0.15] # [p_flip_to_zero,p_flip_to_one]
#     annealing_scheme = torch.Tensor([20,20,15,15,12,12,10,10,10,10])
#     steps_statistics = 10
#     # Make an object from the model and train it
#     bias = True
#     model = BoltzmannMachine(2*n_visible, n_hidden,bias=bias)
#     training.TrainBatch(model,dataset, epochs, learning_rate,noise_levels,steps_statistics,annealing_scheme)
#     print('finished training')

# if __name__ == "__main__":
#     main()
