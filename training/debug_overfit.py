# training/debug_overfit.py
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.auto import tqdm

#Test to see if model is sufficient for bloch equations data 
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]  # Bloch-Equations-Modelling-Project/

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import your model
from models.lightweight_ode_rk4 import NeuralBlochRK4



DATA_DIR   = ROOT / "npz"
TRAIN_NPZ  = DATA_DIR / "train.npz"

# Tiny subset size for overfitting
N_SMALL    = 32

# Training hyperparameters for the overfit test
BATCH_SIZE = 32
EPOCHS     = 300
LR         = 5e-3
WD         = 0.0
HIDDEN     = 64       # small but capable
CLIP_GRAD  = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class BlochDataset(Dataset):
    """
    Dataset wrapper for Bloch .npz files with keys:
      t:  (T,)
      y:  (N, T, 3)
      y0: (N, 3)
      u:  (N, T, 4)
      p:  (N, 5)
    """
    def __init__(self, npz_path: Path):
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Could not find dataset file: {npz_path}")
        print(f"[debug_overfit] Loading data from: {npz_path}")
        data = np.load(npz_path)

        self.y = torch.from_numpy(data["y"]).float()
        self.y0 = torch.from_numpy(data["y0"]).float()
        self.u = torch.from_numpy(data["u"]).float()
        self.p = torch.from_numpy(data["p"]).float()
        self.t = torch.from_numpy(data["t"]).float()

        print(f"[debug_overfit] y shape:  {self.y.shape}")
        print(f"[debug_overfit] y0 shape: {self.y0.shape}")
        print(f"[debug_overfit] u shape:  {self.u.shape}")
        print(f"[debug_overfit] p shape:  {self.p.shape}")
        print(f"[debug_overfit] t shape:  {self.t.shape}")

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return {
            "y":  self.y[idx],
            "y0": self.y0[idx],
            "u":  self.u[idx],
            "p":  self.p[idx],
        }



def overfit_small_subset():
    print(f"[debug_overfit] Using device: {DEVICE}")
    print(f"[debug_overfit] Expecting train.npz at: {TRAIN_NPZ}")

    full_ds = BlochDataset(TRAIN_NPZ)

    if len(full_ds) < N_SMALL:
        raise RuntimeError(
            f"[debug_overfit] Dataset too small: len={len(full_ds)} < N_SMALL={N_SMALL}"
        )

    indices = list(range(N_SMALL))
    small_ds = Subset(full_ds, indices)

    train_loader = DataLoader(
        small_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )

    # Shared time grid
    t_shared = full_ds.t.to(DEVICE)


    model = NeuralBlochRK4(hidden=HIDDEN).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.MSELoss()

    print("[debug_overfit] Starting overfit test on", N_SMALL, "samples")


    first_epoch_loss = None
    last_epoch_loss = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_samples = 0
        grad_norm_example = None

        pbar = tqdm(train_loader, desc=f"[debug_overfit] Epoch {epoch+1}/{EPOCHS}", ncols=100, leave=False)
        for batch in pbar:
            y  = batch["y"].to(DEVICE, non_blocking=True)   # (B,T,3)
            y0 = batch["y0"].to(DEVICE, non_blocking=True)  # (B,3)
            u  = batch["u"].to(DEVICE, non_blocking=True)   # (B,T,4)
            p  = batch["p"].to(DEVICE, non_blocking=True)   # (B,5)

            optimizer.zero_grad()
            yhat = model(y0, t_shared, u, p)  # (B,T,3)

            loss = criterion(yhat, y)
            loss.backward()


            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            if grad_norm_example is None:
                grad_norm_example = float(grad_norm)

            optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            n_samples += bs

            pbar.set_postfix({"loss": f"{loss.item():.4e}", "grad_norm": f"{grad_norm_example:.2e}"})

        epoch_loss = total_loss / max(n_samples, 1)
        if first_epoch_loss is None:
            first_epoch_loss = epoch_loss
        last_epoch_loss = epoch_loss

        print(
            f"[debug_overfit] Epoch {epoch:03d} | "
            f"train_loss = {epoch_loss:.6e} | "
            f"grad_norm ≈ {grad_norm_example:.3e}"
        )

    print("\n[debug_overfit] Finished training on tiny subset.")
    print(f"[debug_overfit] First epoch loss: {first_epoch_loss:.6e}")
    print(f"[debug_overfit] Last epoch loss:  {last_epoch_loss:.6e}")

    if last_epoch_loss < 0.1 * first_epoch_loss:
        print("[debug_overfit] Model CAN overfit this tiny subset (loss dropped by >10x).")
    else:
        print("[debug_overfit] ⚠ Model did NOT significantly overfit this tiny subset.")
        print("               This suggests an optimization or wiring issue (not just model size).")


if __name__ == "__main__":
    overfit_small_subset()
