# train.py
import torch
from torch.utils.data import DataLoader
from data.datasets import BlochNPZ
from models.neural_ode_rk4_fast import NeuralBlochRK4
from training.loop import fit

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    train_ds = BlochNPZ("data/npz/train.npz")
    val_ds   = BlochNPZ("data/npz/val.npz")


    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = NeuralBlochRK4(hidden=128).to(device)
    t_shared = train_ds.t[0].to(device)

    fit(model, train_loader, val_loader, t_shared,
        epochs=80, lr=1e-3, wd=1e-6,
        ckpt_path="checkpoints/best.pt",
        device=device)

if __name__ == "__main__":
    main()
