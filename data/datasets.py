# data/datasets.py
"""
Dataset loader for Bloch Neural ODE .npz files.
Each .npz file should contain:
  t:  (T,) or (N, T)
  y:  (N, T, 3)
  y0: (N, 3)
  u:  (N, T, 4)
  p:  (N, 5)
"""
import numpy as np
import torch
from torch.utils.data import Dataset

class BlochNPZ(Dataset):
    """PyTorch Dataset wrapper for precomputed Bloch-simulator data."""
    def __init__(self, path: str):
        data = np.load(path)
        self.y = torch.from_numpy(data["y"]).float()
        self.y0 = torch.from_numpy(data["y0"]).float()
        self.u = torch.from_numpy(data["u"]).float()
        self.p = torch.from_numpy(data["p"]).float()
        t_data = data["t"]
        if t_data.ndim == 1:
            self.t = torch.from_numpy(t_data).float().unsqueeze(0).repeat(self.y.shape[0], 1)
        else:
            self.t = torch.from_numpy(t_data).float()

    def __len__(self): return self.y.shape[0]

    def __getitem__(self, idx):
        return {
            "t": self.t[idx],   # (T,)
            "y": self.y[idx],   # (T,3)
            "y0": self.y0[idx], # (3,)
            "u": self.u[idx],   # (T,4)
            "p": self.p[idx],   # (5,)
        }
