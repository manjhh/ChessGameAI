import pickle
import torch
from torch.utils.data import Dataset

class ChessDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)  # dạng [(state, policy, value), ...]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, policy, value = self.data[idx]
        # state: tensor [in_channels, 8, 8], policy: tensor [n_moves], value: float
        return torch.tensor(state, dtype=torch.float32), \
               torch.tensor(policy, dtype=torch.float32), \
               torch.tensor(value, dtype=torch.float32)