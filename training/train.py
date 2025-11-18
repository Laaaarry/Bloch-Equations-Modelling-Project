# training/train.py
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# --- Make project root importable when running this file directly ---
ROOT = Path(__file__).resolve().parents[1]  # .. = Bloch-Equations-Modelling-Project
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.loop import fit
from models.lightweight_ode_rk4 import NeuralBlochRK4

# ==========================
# 1. SIMPLE CONFIG SECTION
# ==========================
DATA_DIR = ROOT / "npz"
TRAIN_NPZ = DATA_DIR / "train.npz"
VAL_NPZ   = DATA_DIR / "val.npz"

BATCH_SIZE = 64
EPOCHS     = 80
LR         = 1e-3
WD         = 1e-6
HIDDEN     = 64

CKPT_PATH    = ROOT / "checkpoints" / "best_rk4.pt"
METRICS_PATH = ROOT / "checkpoints" / "metrics_rk4.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# 2. DATASET
# ==========================
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
        data = np.load(npz_path)

        self.y = torch.from_numpy(data["y"]).float()
        self.y0 = torch.from_numpy(data["y0"]).float()
        self.u = torch.from_numpy(data["u"]).float()
        self.p = torch.from_numpy(data["p"]).float()
        self.t = torch.from_numpy(data["t"]).float()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return {
            "y":  self.y[idx],
            "y0": self.y0[idx],
            "u":  self.u[idx],
            "p":  self.p[idx],
        }


# ==========================
# 3. BUILD MODEL + RUN TRAINING
# ==========================
def run_training():
    print(f"Using device: {DEVICE}")

    # --- datasets & loaders ---
    train_ds = BlochDataset(TRAIN_NPZ)
    val_ds   = BlochDataset(VAL_NPZ)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )

    # shared time grid (assume same for train and val)
    t_shared = train_ds.t.to(DEVICE)

    # --- model ---
    model = NeuralBlochRK4(hidden=HIDDEN).to(DEVICE)

    print("Starting training...")
    fit(
        model,
        train_loader,
        val_loader,
        t_shared,
        epochs=EPOCHS,
        lr=LR,
        wd=WD,
        ckpt_path=str(CKPT_PATH),
        metrics_path=str(METRICS_PATH),
        device=DEVICE,
        resume=False,
    )
    print("Training finished.")
    print(f"Best model saved to: {CKPT_PATH}")
    print(f"Metrics CSV saved to: {METRICS_PATH}")


if __name__ == "__main__":
    run_training()
