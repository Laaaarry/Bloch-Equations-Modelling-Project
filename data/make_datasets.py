"""
Dataset generator for Bloch-simulator rollouts to train a Neural ODE.

This script produces .npz files for train/val/test splits. Each sample is a full
time-series rollout produced by your Bloch simulator given:
  - initial state y0
  - control waveforms u(t) = [B1x, B1y, Gx, Gy]
  - static per-sample parameters p = [T1, T2, dB0, B0, gamma]

Saved arrays (default: shared time grid):
  - t:  (T,)              time grid in seconds (shared by all samples)
  - y:  (N, T, 3)         target magnetization trajectories
  - y0: (N, 3)            initial states
  - u:  (N, T, 4)         control waveforms
  - p:  (N, 5)            static parameters per sample

Run:
  python -m data.make_datasets --outdir data/npz --n-train 4000 --n-val 500 --n-test 500 \
      --T 1024 --dt 1e-3 --seed 2025

Notes:
  * If you prefer per-sample time grids (irregular t), set --per-sample-t
    and the script will store t with shape (N, T) instead of (T,).
"""

from __future__ import annotations

import json
import math
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from tqdm.auto import tqdm  # ✅ Progress bar added

# Import your simulator package (installed via `pip install -e .`)
from blochsim import BlochSimulator


# ------------------------------ Utilities ------------------------------ #

def set_seed(seed: int) -> np.random.Generator:
    """
    Create and return a NumPy Generator with a fixed seed.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    numpy.random.Generator
        Seeded random number generator.
    """
    return np.random.default_rng(int(seed))


def ensure_outdir(path: str | Path) -> Path:
    """
    Ensure an output directory exists; create it if missing.

    Parameters
    ----------
    path : str | Path
        Directory path to create if it does not exist.

    Returns
    -------
    Path
        The resolved output directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -------------------------- Parameter Sampling ------------------------- #

def draw_params(rng: np.random.Generator) -> Dict[str, float]:
    """
    Sample physically reasonable Bloch parameters for one rollout.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    dict
        A dictionary with keys:
          - T1 : float   in seconds (longitudinal relaxation)
          - T2 : float   in seconds (transverse relaxation)
          - dB0: float   static off-resonance (arbitrary units consistent with sim)
          - B0 : float   main field strength (same units as simulator expects)
          - gamma: float gyromagnetic ratio (keep 1.0 to normalize units)
    """
    return dict(
        T1=float(rng.uniform(1.5, 4.0)),
        T2=float(rng.uniform(0.6, 2.0)),
        dB0=float(rng.uniform(-0.5, 0.5)),
        B0=float(rng.uniform(1.5, 3.0)),
        gamma=1.0,
    )

# --------------------------- Control Programming ------------------------ #

def program_controls_and_run(
    sim: BlochSimulator,
    T: int,
    dt: float,
    rng: np.random.Generator,
    rf_families: tuple[str, ...] = ("rect", "sinc", "gaussian", "windowed_sinc", "trapezoid"),
) -> tuple[np.ndarray, str]:
    """
    Program one RF + one gradient lobe, evolve the simulator, and return controls.
    Now supports multiple RF families. Returns (u, rf_type).

    u[t] = [B1x, B1y, Gx, Gy], shape (T, 4)
    rf_type: the string name of the chosen RF family
    """
    u = np.zeros((T, 4), dtype=np.float32)

    # --- Random RF parameters (shared across families) ---
    flip  = rng.uniform(math.radians(10), math.radians(120))
    phase = rng.uniform(0, 2 * math.pi)
    B1amp = rng.uniform(2.5, 5.0)

    # --- Choose and arm an RF pulse family on the simulator ---
    rf_type = rng.choice(rf_families)

    # Dispatch to your simulator methods (you added these)
    dispatch = {
        "rect":           lambda: sim.RF_pulse_rect(angle=flip, phase=phase, B1_amplitude=B1amp),
        "sinc":           lambda: sim.RF_pulse_sinc(angle=flip, phase=phase, B1_amplitude=B1amp, n_lobes=4),
        "gaussian":       lambda: sim.RF_pulse_gaussian(angle=flip, phase=phase, B1_amplitude=B1amp, sigma_frac=0.20),
        "windowed_sinc":  lambda: sim.RF_pulse_windowed_sinc(angle=flip, phase=phase, B1_amplitude=B1amp, n_lobes=4),
        "trapezoid":      lambda: sim.RF_pulse_trapezoid(angle=flip, phase=phase, B1_amplitude=B1amp, ramp_frac=0.10),
    }
    if rf_type not in dispatch:
        raise ValueError(f"Unknown rf_type '{rf_type}'. Supported: {list(dispatch.keys())}")
    dispatch[rf_type]()  # arm the RF

    # --- One gradient lobe (unchanged) ---
    phi   = rng.uniform(0.0, math.pi)       # gradient area (internal scaling in sim)
    theta = rng.uniform(0.0, 2 * math.pi)   # gradient direction
    sim.apply_gradient(phase_diff=phi, direction_angle=theta)

    # --- Drive system & record realized controls ---
    for t_idx in range(T):
        B1x = sim.B1 * math.cos(sim.B1_freq * sim.time - sim.phi1)
        B1y = -sim.B1 * math.sin(sim.B1_freq * sim.time - sim.phi1)
        u[t_idx, 0] = B1x
        u[t_idx, 1] = B1y
        u[t_idx, 2] = sim.Gx
        u[t_idx, 3] = sim.Gy
        sim.step(dt)

    sim.Gx = 0.0
    sim.Gy = 0.0
    return u, rf_type


# --------------------------- Single Rollout ----------------------------- #

def simulate_one(
    T: int,
    dt: float,
    rng: np.random.Generator,
    per_sample_t: bool = False,
    rf_families: tuple[str, ...] = ("rect", "sinc", "gaussian", "windowed_sinc", "trapezoid"),
) -> Dict[str, np.ndarray]:
    """
    Simulate one full trajectory sample. Now also returns 'rf_type' (string).
    """
    # --- Sample parameters and build simulator ---
    p = draw_params(rng)
    sim = BlochSimulator(gamma=p["gamma"], B0=p["B0"])

    if rng.random() < 0.5:
        sim.setup_single_spin()
        sim.isochromats[0].dB0 = p["dB0"]
        sim.isochromats[0].T1  = p["T1"]
        sim.isochromats[0].T2  = p["T2"]
    else:
        n = rng.integers(7, 17)
        spread = rng.uniform(0.1, 0.6)
        sim.setup_inhomogeneous(n_spins=int(n), spread=float(spread))
        for iso in sim.isochromats:
            iso.T1 = p["T1"]
            iso.T2 = p["T2"]

    sim.reset()

    t = np.arange(T, dtype=np.float32) * float(dt)

    # Program controls and run (now returns rf_type)
    u, rf_type = program_controls_and_run(sim, T=T, dt=dt, rng=rng, rf_families=rf_families)

    y = np.array(
        [np.sum(state["magnetization"], axis=0) for state in sim.history],
        dtype=np.float32,
    )
    y0 = y[0].astype(np.float32)
    p_vec = np.array([p["T1"], p["T2"], p["dB0"], p["B0"], p["gamma"]], dtype=np.float32)

    return dict(t=t, y=y, y0=y0, u=u, p=p_vec, rf_type=np.array(rf_type))


# --------------------------- Split Writers ------------------------------ #

def write_split(
    outdir: str | Path,
    n_train: int,
    n_val: int,
    n_test: int,
    T: int,
    dt: float,
    seed: int = 2025,
    per_sample_t: bool = False,
    rf_families: tuple[str, ...] = ("rect", "sinc", "gaussian", "windowed_sinc", "trapezoid"),
) -> None:
    """
    Generate train/val/test splits and write compressed NPZ files.
    Adds 'rf_type' (N,) string array to each split.
    """
    outdir = ensure_outdir(outdir)
    rng = set_seed(seed)

    t_shared = np.arange(T, dtype=np.float32) * float(dt)

    def make_n(n: int, split_name: str):
        Ts, Ys, Y0s, Us, Ps, RFs = [], [], [], [], [], []
        for _ in tqdm(range(n), desc=f"[make_datasets] Generating {split_name} set", ncols=80):
            s = simulate_one(T=T, dt=dt, rng=rng, per_sample_t=per_sample_t, rf_families=rf_families)
            Ts.append(s["t"])
            Ys.append(s["y"])
            Y0s.append(s["y0"])
            Us.append(s["u"])
            Ps.append(s["p"])
            RFs.append(s["rf_type"])  # string
        T_arr = np.stack(Ts, axis=0)
        Y_arr = np.stack(Ys, axis=0)
        Y0_arr = np.stack(Y0s, axis=0)
        U_arr = np.stack(Us, axis=0)
        P_arr = np.stack(Ps, axis=0)
        RF_arr = np.array([rf.item() if hasattr(rf, "item") else str(rf) for rf in RFs], dtype=np.str_)
        return T_arr, Y_arr, Y0_arr, U_arr, P_arr, RF_arr

    t_tr, y_tr, y0_tr, u_tr, p_tr, rf_tr = make_n(n_train, "training")
    t_va, y_va, y0_va, u_va, p_va, rf_va = make_n(n_val, "validation")
    t_te, y_te, y0_te, u_te, p_te, rf_te = make_n(n_test, "test")

    def save_split(fname: str, t_arr, y_arr, y0_arr, u_arr, p_arr, rf_arr):
        fpath = outdir / fname
        if per_sample_t:
            np.savez_compressed(fpath, t=t_arr, y=y_arr, y0=y0_arr, u=u_arr, p=p_arr, rf_type=rf_arr)
        else:
            np.savez_compressed(fpath, t=t_shared, y=y_arr, y0=y0_arr, u=u_arr, p=p_arr, rf_type=rf_arr)

    save_split("train.npz", t_tr, y_tr, y0_tr, u_tr, p_tr, rf_tr)
    save_split("val.npz",   t_va, y_va, y0_va, u_va, p_va, rf_va)
    save_split("test.npz",  t_te, y_te, y0_te, u_te, p_te, rf_te)

    meta = dict(
        T=T,
        dt=float(dt),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=int(seed),
        per_sample_t=bool(per_sample_t),
        rf_families=list(rf_families),
    )
    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[make_datasets] Wrote splits to: {outdir.resolve()}")
    print(f"[make_datasets] Shapes (train): y={y_tr.shape}, u={u_tr.shape}, p={p_tr.shape}, rf={rf_tr.shape}")


# ----------------------------- CLI Entrypoint --------------------------- #

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate Bloch Neural-ODE datasets (.npz).")
    ap.add_argument("--outdir", type=str, default="data/npz", help="Output directory for .npz files.")
    ap.add_argument("--n-train", type=int, default=4000, help="Number of training samples.")
    ap.add_argument("--n-val", type=int, default=500, help="Number of validation samples.")
    ap.add_argument("--n-test", type=int, default=500, help="Number of test samples.")
    ap.add_argument("--T", type=int, default=1024, help="Number of time steps per sample.")
    ap.add_argument("--dt", type=float, default=1e-3, help="Time step (seconds).")
    ap.add_argument("--seed", type=int, default=2025, help="Random seed.")
    ap.add_argument("--per-sample-t", action="store_true", help="Store per-sample time grids.")
    ap.add_argument(
        "--rf-families",
        type=str,
        default="rect,sinc,gaussian,windowed_sinc,trapezoid",
        help="Comma-separated RF families to sample from per rollout.",
    )
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    rf_families = tuple([s.strip() for s in args.rf_families.split(",") if s.strip()])
    write_split(
        outdir=args.outdir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        T=args.T,
        dt=args.dt,
        seed=args.seed,
        per_sample_t=bool(args.per_sample_t),
        rf_families=rf_families,
    )

if __name__ == "__main__":  
    main()