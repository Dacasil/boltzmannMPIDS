from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class BasicEncoderDataset(CustomDataset):
    def __init__(self, n_visible, device="cpu"):
        """
        reps: control parameter for correlation
        """
        # create correlated data
        correlated_patterns = torch.zeros(n_visible, 2 * n_visible)
        for i in range(n_visible):
            correlated_patterns[i, i] = 1
            correlated_patterns[i, n_visible + i] = 1
        patterns = correlated_patterns
        # binary to bipolar
        # patterns = 2*patterns-1
        # finish
        self.data = patterns
        super().__init__(self.data)


class CorrelatedEncoderDataset(CustomDataset):
    def __init__(self, n_visible, reps, device="cpu"):
        """
        reps: control parameter for correlation
        """
        # generate all possible combinations of patterns
        loc_a = torch.arange(n_visible)
        indices = torch.combinations(loc_a)
        # create a pattern for each
        patterns = torch.zeros((len(indices), 2 * n_visible))
        for i, pair in enumerate(indices):
            x, y = pair
            patterns[i, x] = 1
            patterns[i, n_visible + y] = 1
        # create correlated data
        correlated_patterns = torch.zeros(n_visible, 2 * n_visible)
        for i in range(n_visible):
            correlated_patterns[i, i] = 1
            correlated_patterns[i, n_visible + i] = 1
        # concat correlated_patterns
        correlated_patterns = correlated_patterns.repeat(reps, 1)
        patterns = torch.concat((patterns, correlated_patterns))
        # binary to bipolar
        patterns = 2 * patterns - 1
        # finish
        self.data = patterns
        super().__init__(self.data)
