# training/eval_inference.py
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]  # .. = Bloch-Equations-Modelling-Project
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
            raise FileNotFoundError(f"Data file not found: {npz_path}")
        print(f"[eval_inference] Loading data from: {npz_path}")
        data = np.load(npz_path)

        self.y = torch.from_numpy(data["y"]).float()
        self.y0 = torch.from_numpy(data["y0"]).float()
        self.u = torch.from_numpy(data["u"]).float()
        self.p = torch.from_numpy(data["p"]).float()
        self.t = torch.from_numpy(data["t"]).float()  # shared time grid

        print(f"[eval_inference] y shape:  {self.y.shape}")
        print(f"[eval_inference] y0 shape: {self.y0.shape}")
        print(f"[eval_inference] u shape:  {self.u.shape}")
        print(f"[eval_inference] p shape:  {self.p.shape}")
        print(f"[eval_inference] t shape:  {self.t.shape}")

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
# 2. Helper: load model from checkpoint
# ==========================
def load_trained_model(model_cls, ckpt_path: Path, hidden: int = 64, device: torch.device = DEVICE):
    """
    model_cls: the class object for your model (e.g., NeuralBlochRK4)
    ckpt_path: path to a checkpoint saved by train.py
    hidden:    hidden size to construct the model with
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    print(f"[eval_inference] Loading checkpoint: {ckpt_path}")

    model = model_cls(hidden=hidden).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ==========================
# 3. Evaluate on test set
# ==========================
@torch.no_grad()
def evaluate_on_test(
    model_cls,
    hidden: int = 64,
    data_dir: Path | None = None,
    ckpt_path: Path | None = None,
    batch_size: int = 64,
    device: torch.device = DEVICE,
):
    """
    Run inference on test.npz using a trained model.

    Args:
        model_cls:   model class to instantiate (takes hidden=hidden)
        hidden:      hidden size to pass to model_cls
        data_dir:    directory containing train/val/test npz (default: ROOT/'npz')
        ckpt_path:   path to checkpoint (default: ROOT/'checkpoints'/'best_rk4.pt')
        batch_size:  batch size for inference
        device:      torch.device
    """
    if data_dir is None:
        data_dir = ROOT / "npz"
    if ckpt_path is None:
        ckpt_path = ROOT / "checkpoints" / "best_rk4.pt"

    print(f"[eval_on_test] Using device: {device}")
    test_npz = Path(data_dir) / "test.npz"

    # dataset & loader
    test_ds = BlochDataset(test_npz)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    t_shared = test_ds.t.to(device)

    # model
    model = load_trained_model(model_cls, ckpt_path, hidden=hidden, device=device)
    criterion = nn.MSELoss(reduction="sum")  # we'll normalize manually

    total_loss = 0.0
    total_samples = 0
    total_loss_comp = torch.zeros(3, device=device)

    pbar = tqdm(test_loader, desc="[eval_on_test] Running inference", ncols=100)
    for batch in pbar:
        y  = batch["y"].to(device, non_blocking=True)   # (B,T,3)
        y0 = batch["y0"].to(device, non_blocking=True)  # (B,3)
        u  = batch["u"].to(device, non_blocking=True)   # (B,T,4)
        p  = batch["p"].to(device, non_blocking=True)   # (B,5)

        yhat = model(y0, t_shared, u, p)

        batch_loss = criterion(yhat, y)
        total_loss += batch_loss.item()
        total_samples += y.numel()

        diff = (yhat - y).reshape(-1, 3)
        total_loss_comp += (diff ** 2).sum(dim=0)

        pbar.set_postfix({"batch_MSE": f"{(batch_loss.item()/y.numel()):.4e}"})

    mean_mse = total_loss / total_samples
    mean_mse_comp = (total_loss_comp / total_samples).cpu().numpy()

    print("\n[eval_on_test] ===== Test results =====")
    print(f"Global MSE: {mean_mse:.6e}")
    print(f"MSE Mx: {mean_mse_comp[0]:.6e}")
    print(f"MSE My: {mean_mse_comp[1]:.6e}")
    print(f"MSE Mz: {mean_mse_comp[2]:.6e}")

    return {
        "global_mse": float(mean_mse),
        "mse_components": mean_mse_comp,
    }


# ==========================
# 4. Analytic Bloch solution (simple case)
# ==========================
def bloch_analytic_solution(t, y0, p, M_eq=1.0):
    """
    Analytic Bloch solution for simple case:
      - no RF or gradients (u = 0)
      - constant B0 along z
      - relaxation with T1, T2
      - precession around z: omega = gamma * (B0 + dB0)

    t:   (T,)
    y0:  (B,3)
    p:   (B,5)  [T1, T2, dB0, B0, gamma]
    """
    T = t.shape[0]
    t = t.view(1, T, 1)

    T1 = p[:, 0].view(-1, 1, 1)
    T2 = p[:, 1].view(-1, 1, 1)
    dB0 = p[:, 2].view(-1, 1, 1)
    B0  = p[:, 3].view(-1, 1, 1)
    gamma = p[:, 4].view(-1, 1, 1)

    Mx0 = y0[:, 0].view(-1, 1, 1)
    My0 = y0[:, 1].view(-1, 1, 1)
    Mz0 = y0[:, 2].view(-1, 1, 1)

    Mxy0 = torch.sqrt(Mx0**2 + My0**2 + 1e-12)
    phi0 = torch.atan2(My0, Mx0)

    E1 = torch.exp(-t / (T1 + 1e-12))
    E2 = torch.exp(-t / (T2 + 1e-12))

    omega = gamma * (B0 + dB0)
    phase = omega * t + phi0

    Mx = Mxy0 * E2 * torch.cos(phase)
    My = Mxy0 * E2 * torch.sin(phase)
    Mz = M_eq + (Mz0 - M_eq) * E1

    y = torch.cat([Mx, My, Mz], dim=-1)  # (B,T,3)
    return y


# ==========================
# 5. Compare model vs analytic Bloch
# ==========================
@torch.no_grad()
def evaluate_against_analytic(
    model_cls,
    hidden: int = 64,
    ckpt_path: Path | None = None,
    B: int = 16,
    T: int = 256,
    dt: float = 1e-3,
    device: torch.device = DEVICE,
):
    """
    Compare the trained model to an analytic Bloch solution for a simple case
    (no RF, no gradients, relaxation + precession).

    Args:
        model_cls: model class to instantiate
        hidden:    hidden size for model
        ckpt_path: checkpoint path (default: ROOT/'checkpoints'/'best_rk4.pt')
        B:         number of random test samples
        T:         number of timesteps
        dt:        time step
        device:    torch.device
    """
    if ckpt_path is None:
        ckpt_path = ROOT / "checkpoints" / "best_rk4.pt"

    print(f"[analytic_eval] Using device: {device}")
    t = torch.arange(T, dtype=torch.float32, device=device) * dt  # (T,)

    # Set global seed
    torch.manual_seed(123)

    # Random initial magnetization (roughly unit magnitude)
    y0 = torch.randn(B, 3, device=device)
    y0 = y0 / (y0.norm(dim=-1, keepdim=True) + 1e-8)

    # Random Bloch parameters in a reasonable range (uniform on device)
    T1 = torch.empty(B, device=device).uniform_(1.5, 4.0)
    T2 = torch.empty(B, device=device).uniform_(0.6, 2.0)
    dB0 = torch.empty(B, device=device).uniform_(-0.5, 0.5)
    B0  = torch.empty(B, device=device).uniform_(1.5, 3.0)
    gamma = torch.ones(B, device=device)


    p = torch.stack([T1, T2, dB0, B0, gamma], dim=-1)  # (B,5)
    u = torch.zeros(B, T, 4, device=device)  # controls = 0

    model = load_trained_model(model_cls, ckpt_path, hidden=hidden, device=device)

    y_model = model(y0, t, u, p)
    y_analytic = bloch_analytic_solution(t, y0, p, M_eq=1.0)

    mse_global = torch.mean((y_model - y_analytic) ** 2).item()
    mse_comp = torch.mean((y_model - y_analytic) ** 2, dim=(0, 1)).cpu().numpy()

    print("\n[analytic_eval] ===== Model vs Analytic Bloch (simple case) =====")
    print(f"Global MSE: {mse_global:.6e}")
    print(f"MSE Mx: {mse_comp[0]:.6e}")
    print(f"MSE My: {mse_comp[1]:.6e}")
    print(f"MSE Mz: {mse_comp[2]:.6e}")

    return {
        "global_mse": float(mse_global),
        "mse_components": mse_comp,
    }


# ==========================
# 6. Optional CLI demo
# ==========================
if __name__ == "__main__":
    #from models.lightweight_ode_rk4 import NeuralBlochRK4
    from models.neural_bloch_gru import NeuralBlochGRU

    print("Evaluating on test set with NeuralBlochGRU")
    evaluate_on_test(NeuralBlochGRU, ckpt_path=ROOT / "checkpoints" / "best_gru.pt", hidden=64)

    print("\nEvaluating vs analytic Bloch with NeuralBlochGRU")
    evaluate_against_analytic(NeuralBlochGRU, ckpt_path=ROOT / "checkpoints" / "best_gru.pt", hidden=64)