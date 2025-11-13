from __future__ import annotations

import json, math, argparse
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from tqdm.auto import tqdm

from blochsim import BlochSimulator


# ------------------------------ Utilities ------------------------------ #

def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))

def ensure_outdir(path: str | Path) -> Path:
    """
    Ensure an output directory exists; create it if missing.
    Prints the absolute resolved path for debugging, and fails loudly if needed.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create output directory: {p}\n{e}") from e
    print(f"[make_datasets] Using output directory: {p.resolve()}")
    return p


# -------------------------- Parameter Sampling ------------------------- #

def draw_params(rng: np.random.Generator) -> Dict[str, float]:
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
    rf_families: tuple[str, ...] = ("rect","sinc","gaussian","windowed_sinc","trapezoid"),
) -> tuple[np.ndarray, str]:
    u = np.zeros((T, 4), dtype=np.float32)

    flip  = rng.uniform(math.radians(10), math.radians(120))
    phase = rng.uniform(0, 2 * math.pi)
    B1amp = rng.uniform(2.5, 5.0)

    rf_type = rng.choice(rf_families)
    dispatch = {
        "rect":          lambda: sim.RF_pulse_rect(angle=flip, phase=phase, B1_amplitude=B1amp),
        "sinc":          lambda: sim.RF_pulse_sinc(angle=flip, phase=phase, B1_amplitude=B1amp, n_lobes=4),
        "gaussian":      lambda: sim.RF_pulse_gaussian(angle=flip, phase=phase, B1_amplitude=B1amp, sigma_frac=0.20),
        "windowed_sinc": lambda: sim.RF_pulse_windowed_sinc(angle=flip, phase=phase, B1_amplitude=B1amp, n_lobes=4),
        "trapezoid":     lambda: sim.RF_pulse_trapezoid(angle=flip, phase=phase, B1_amplitude=B1amp, ramp_frac=0.10),
    }
    if rf_type not in dispatch:
        raise ValueError(f"Unknown rf_type '{rf_type}'. Supported: {list(dispatch.keys())}")
    dispatch[rf_type]()

    phi   = rng.uniform(0.0, math.pi)
    theta = rng.uniform(0.0, 2 * math.pi)
    sim.apply_gradient(phase_diff=phi, direction_angle=theta)

    for t_idx in range(T):
        B1x = sim.B1 * math.cos(sim.B1_freq * sim.time - sim.phi1)
        B1y = -sim.B1 * math.sin(sim.B1_freq * sim.time - sim.phi1)
        u[t_idx, 0] = B1x
        u[t_idx, 1] = B1y
        u[t_idx, 2] = sim.Gx
        u[t_idx, 3] = sim.Gy
        sim.step(dt)

    sim.Gx = 0.0; sim.Gy = 0.0
    return u, rf_type


# --------------------------- Single Rollout ----------------------------- #

def simulate_one(
    T: int,
    dt: float,
    rng: np.random.Generator,
    per_sample_t: bool = False,
    rf_families: tuple[str, ...] = ("rect","sinc","gaussian","windowed_sinc","trapezoid"),
) -> Dict[str, np.ndarray]:
    p = draw_params(rng)
    sim = BlochSimulator(gamma=p["gamma"], B0=p["B0"])

    if rng.random() < 0.5:
        sim.setup_single_spin()
        sim.isochromats[0].dB0 = p["dB0"]
        sim.isochromats[0].T1  = p["T1"]
        sim.isochromats[0].T2  = p["T2"]
    else:
        n = rng.integers(7, 17); spread = rng.uniform(0.1, 0.6)
        sim.setup_inhomogeneous(n_spins=int(n), spread=float(spread))
        for iso in sim.isochromats:
            iso.T1 = p["T1"]; iso.T2 = p["T2"]

    sim.reset()
    t = np.arange(T, dtype=np.float32) * float(dt)

    u, rf_type = program_controls_and_run(sim, T=T, dt=dt, rng=rng, rf_families=rf_families)

    y = np.array([np.sum(state["magnetization"], axis=0) for state in sim.history], dtype=np.float32)
    y0 = y[0].astype(np.float32)
    p_vec = np.array([p["T1"], p["T2"], p["dB0"], p["B0"], p["gamma"]], dtype=np.float32)

    return dict(t=t, y=y, y0=y0, u=u, p=p_vec, rf_type=str(rf_type))


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
    rf_families: tuple[str, ...] = ("rect","sinc","gaussian","windowed_sinc","trapezoid"),
) -> None:
    """
    Generate train/val/test splits and write compressed NPZ files.
    Adds 'rf_type' (N,) string array to each split. Uses tqdm for progress and
    surfaces the first failure clearly.
    """
    outdir = ensure_outdir(outdir)
    rng = set_seed(seed)
    t_shared = np.arange(T, dtype=np.float32) * float(dt)

    def make_n(n: int, split_name: str):
        if n == 0:
            print(f"[make_datasets] Skipping empty split: {split_name} (n=0)")
            return None
        Ts, Ys, Y0s, Us, Ps, RFs = [], [], [], [], [], []
        bar = tqdm(range(n), desc=f"[make_datasets] {split_name}", ncols=80)
        for i in bar:
            try:
                s = simulate_one(T=T, dt=dt, rng=rng, per_sample_t=per_sample_t, rf_families=rf_families)
            except Exception as e:
                bar.close()
                raise RuntimeError(f"Error while generating {split_name} sample {i}/{n}: {e}") from e
            Ts.append(s["t"]); Ys.append(s["y"]); Y0s.append(s["y0"]); Us.append(s["u"]); Ps.append(s["p"]); RFs.append(s["rf_type"])
        T_arr = np.stack(Ts, axis=0)
        Y_arr = np.stack(Ys, axis=0)
        Y0_arr = np.stack(Y0s, axis=0)
        U_arr = np.stack(Us, axis=0)
        P_arr = np.stack(Ps, axis=0)
        RF_arr = np.array(RFs, dtype="<U24")  # plain unicode array
        return T_arr, Y_arr, Y0_arr, U_arr, P_arr, RF_arr

    res_tr = make_n(n_train, "training")
    res_va = make_n(n_val,   "validation")
    res_te = make_n(n_test,  "test")

    def save_split(fname: str, pack):
        if pack is None:  # allow 0-sized splits
            return
        t_arr, y_arr, y0_arr, u_arr, p_arr, rf_arr = pack
        fpath = outdir / fname
        if per_sample_t:
            np.savez_compressed(fpath, t=t_arr, y=y_arr, y0=y0_arr, u=u_arr, p=p_arr, rf_type=rf_arr)
        else:
            np.savez_compressed(fpath, t=t_shared, y=y_arr, y0=y0_arr, u=u_arr, p=p_arr, rf_type=rf_arr)
        print(f"[make_datasets] Wrote {fname} → {fpath.resolve()}")

    save_split("train.npz", res_tr)
    save_split("val.npz",   res_va)
    save_split("test.npz",  res_te)

    meta = dict(T=T, dt=float(dt), n_train=n_train, n_val=n_val, n_test=n_test,
                seed=int(seed), per_sample_t=bool(per_sample_t), rf_families=list(rf_families))
    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[make_datasets] Meta → {(outdir / 'meta.json').resolve()}")


# ----------------------------- CLI Entrypoint --------------------------- #

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate Bloch Neural-ODE datasets (.npz).")
    ap.add_argument("--outdir", type=str, default="data/npz", help="Output directory for .npz files.")
    ap.add_argument("--n-train", type=int, default=4000, help="Number of training samples.")
    ap.add_argument("--n-val",   type=int, default=500,  help="Number of validation samples.")
    ap.add_argument("--n-test",  type=int, default=500,  help="Number of test samples.")
    ap.add_argument("--T", type=int, default=1024, help="Number of time steps per sample.")
    ap.add_argument("--dt", type=float, default=1e-3, help="Time step (seconds).")
    ap.add_argument("--seed", type=int, default=2025, help="Random seed.")
    ap.add_argument("--per-sample-t", action="store_true", help="Store per-sample time grids.")
    ap.add_argument("--rf-families", type=str,
                    default="rect,sinc,gaussian,windowed_sinc,trapezoid",
                    help="Comma-separated RF families to sample from per rollout.")
    return ap

def main() -> None:
    args = build_argparser().parse_args()
    rf_families = tuple([s.strip() for s in args.rf_families.split(",") if s.strip()])
    write_split(
        outdir=args.outdir, n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        T=args.T, dt=args.dt, seed=args.seed, per_sample_t=bool(args.per_sample_t),
        rf_families=rf_families,
    )


# ------------------------- Explicit in-code run ------------------------- #
# If you want dataset creation to happen "explicitly in code" without any
# command-line, call generate_dataset_inline() from anywhere (or set AUTO_RUN=True).

def generate_dataset_inline():
    """Create a tiny dataset explicitly (no CLI)."""
    print("[make_datasets] Running generate_dataset_inline() …")
    write_split(
        outdir=ensure_outdir("data/npz_small_inline"),
        n_train=5, n_val=1, n_test=1,
        T=128, dt=1e-3, seed=123, per_sample_t=False,
        rf_families=("rect","sinc"),  # add more when your RF methods are ready
    )

AUTO_RUN = False  # set to True to run immediately on import

if __name__ == "__main__":
    # EITHER: explicit in-code generation …
    if AUTO_RUN:
        generate_dataset_inline()
    # … OR: standard CLI entrypoint
    else:
        main()
