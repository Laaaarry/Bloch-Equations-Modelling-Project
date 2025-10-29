# training/loop.py
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def train_one_epoch(model, loader, t_shared, optimizer, criterion, device, epoch=None, total_epochs=None):
    model.train()
    total_loss, n = 0.0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", leave=False, ncols=90)
    for batch in pbar:
        # --- move to GPU here ---
        y, y0, u, p = [batch[k].to(device, non_blocking=True) for k in ("y","y0","u","p")]
        # shared time grid is already on device (t_shared.to(device) in train.py)

        optimizer.zero_grad()
        yhat = model(y0, t_shared, u, p)
        loss = criterion(yhat, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        pbar.set_postfix({"batch_loss": f"{loss.item():.4e}"})
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, t_shared, criterion, device, epoch=None, total_epochs=None):
    model.eval()
    total_loss, n = 0.0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]", leave=False, ncols=90)
    for batch in pbar:
        y, y0, u, p = [batch[k].to(device, non_blocking=True) for k in ("y","y0","u","p")]
        yhat = model(y0, t_shared, u, p)
        loss = criterion(yhat, y)
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        pbar.set_postfix({"batch_loss": f"{loss.item():.4e}"})
    return total_loss / n


def fit(model, train_loader, val_loader, t_shared, epochs=80, lr=1e-3, wd=1e-6,
        ckpt_path="checkpoints/best.pt", device="cpu"):
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()
    best_val = float("inf")

    epoch_bar = tqdm(range(epochs), desc="Training progress", ncols=100)
    for epoch in epoch_bar:
        tr = train_one_epoch(model, train_loader, t_shared, optimizer, criterion, device,
                             epoch=epoch, total_epochs=epochs)
        va = evaluate(model, val_loader, t_shared, criterion, device,
                      epoch=epoch, total_epochs=epochs)
        epoch_bar.set_postfix({"train": f"{tr:.4e}", "val": f"{va:.4e}"})
        print(f"Epoch {epoch:03d} | Train {tr:.6f} | Val {va:.6f}")

        if va < best_val:
            best_val = va
            torch.save({"model": model.state_dict(), "val": va, "epoch": epoch}, ckpt_path)
