from pathlib import Path
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train_one_epoch(
    model,
    loader,
    t_shared,
    optimizer,
    criterion,
    device,
    epoch=None,
    total_epochs=None,
    lambda_phys: float = 1.0,  # NEW: weight for physics loss
):
    model.train()
    total_loss, total_elems = 0.0, 0
    total_comp = torch.zeros(3, device=device)  # Mx, My, Mz

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", leave=False, ncols=90)
    for batch in pbar:
        y, y0, u, p = [batch[k].to(device, non_blocking=True) for k in ("y", "y0", "u", "p")]

        optimizer.zero_grad()

        # --- supervised term ---
        yhat = model(y0, t_shared, u, p)     # (B,T,3)
        data_loss = criterion(yhat, y)       # supervised MSE

        # --- physics-informed term (unsupervised) ---
        # Only works for models that implement physics_residual
        if hasattr(model, "physics_residual"):
            residual = model.physics_residual(y0, t_shared, u, p)  # (B,T,3)
            physics_loss = torch.mean(residual ** 2)
        else:
            physics_loss = torch.tensor(0.0, device=device)

        # --- total loss ---
        loss = data_loss + lambda_phys * physics_loss

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # logging: you can choose to log data_loss or total loss
        batch_elems = y.numel()  # B*T*3
        total_loss += data_loss.item() * batch_elems
        total_elems += batch_elems

        diff2 = (yhat - y) ** 2
        total_comp += diff2.sum(dim=(0, 1))

        pbar.set_postfix({
            "data": f"{data_loss.item():.4e}",
            "phys": f"{physics_loss.item():.4e}",
        })

    mean_loss = total_loss / max(total_elems, 1)
    num_positions = total_elems / 3.0 if total_elems > 0 else 1.0
    mean_comp = total_comp / num_positions  # (3,)

    return mean_loss, mean_comp.detach().cpu()



@torch.no_grad()
def evaluate(model, loader, t_shared, criterion, device, epoch=None, total_epochs=None):
    model.eval()
    total_loss, total_elems = 0.0, 0
    total_comp = torch.zeros(3, device=device)  # Mx, My, Mz

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]", leave=False, ncols=90)
    for batch in pbar:
        y, y0, u, p = [batch[k].to(device, non_blocking=True) for k in ("y", "y0", "u", "p")]
        yhat = model(y0, t_shared, u, p)
        loss = criterion(yhat, y)

        batch_elems = y.numel()
        total_loss += loss.item() * batch_elems
        total_elems += batch_elems

        diff2 = (yhat - y) ** 2
        total_comp += diff2.sum(dim=(0, 1))

        pbar.set_postfix({"batch_loss": f"{loss.item():.4e}"})

    mean_loss = total_loss / max(total_elems, 1)
    num_positions = total_elems / 3.0 if total_elems > 0 else 1.0
    mean_comp = total_comp / num_positions  # (3,)

    return mean_loss, mean_comp.detach().cpu()


def fit(
    model,
    train_loader,
    val_loader,
    t_shared,
    epochs=80,
    lr=1e-3,
    wd=1e-6,
    ckpt_path="checkpoints/best.pt",
    metrics_path=None,
    device="cpu",
    resume=False,
):
    """
    Train the model and optionally resume from an existing checkpoint.

    If resume=True and ckpt_path exists, this will:
      - load model weights
      - load optimizer state (if present)
      - restore best_val
      - resume from epoch = last_epoch + 1

    Otherwise, it trains from scratch.
    """
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional CSV logging setup
    metrics_file = None
    write_header = False
    if metrics_path is not None:
        metrics_file = Path(metrics_path)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        write_header = not metrics_file.exists()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()

    best_val = float("inf")
    start_epoch = 0

    # --------- resume logic ----------
    if resume and ckpt_path.exists():
        print(f"[loop.fit] Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                print("[loop.fit] Loaded optimizer state from checkpoint.")
            except Exception as e:
                print(f"[loop.fit] Warning: could not load optimizer state: {e}")
        best_val = ckpt.get("val", float("inf"))
        last_epoch = ckpt.get("epoch", -1)
        start_epoch = last_epoch + 1
        print(f"[loop.fit] Last epoch in checkpoint: {last_epoch} → starting from epoch {start_epoch}")
    else:
        if resume:
            print(f"[loop.fit] No checkpoint found at {ckpt_path}, starting from scratch.")
        else:
            print("[loop.fit] Starting training from scratch.")

    # Main training loop
    epoch_bar = tqdm(range(start_epoch, epochs), desc="Training progress", ncols=100)
    for epoch in epoch_bar:
        tr, tr_comp = train_one_epoch(
            model, train_loader, t_shared, optimizer, criterion, device,
            epoch=epoch, total_epochs=epochs
        )
        va, va_comp = evaluate(
            model, val_loader, t_shared, criterion, device,
            epoch=epoch, total_epochs=epochs
        )

        # unpack per-component losses
        tr_Mx, tr_My, tr_Mz = tr_comp.tolist()
        va_Mx, va_My, va_Mz = va_comp.tolist()

        epoch_bar.set_postfix({
            "train": f"{tr:.4e}",
            "val":   f"{va:.4e}",
            "Mx":    f"{va_Mx:.2e}",
            "My":    f"{va_My:.2e}",
            "Mz":    f"{va_Mz:.2e}",
        })

        print(
            f"Epoch {epoch:03d} | "
            f"Train {tr:.6f} (Mx {tr_Mx:.3e}, My {tr_My:.3e}, Mz {tr_Mz:.3e}) | "
            f"Val {va:.6f} (Mx {va_Mx:.3e}, My {va_My:.3e}, Mz {va_Mz:.3e})"
        )

        # --- CSV logging (still only scalar losses, unless you want more columns) ---
        if metrics_file is not None:
            with metrics_file.open("a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["epoch", "train_loss", "val_loss"])
                    write_header = False
                writer.writerow([epoch, tr, va])

        # --- save best checkpoint (still based on scalar val loss) ---
        if va < best_val:
            best_val = va
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val": va,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"[loop.fit] Saved new best checkpoint at epoch {epoch} (val={va:.6e})")

