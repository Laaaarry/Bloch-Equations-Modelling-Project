import numpy as np
import matplotlib.pyplot as plt
from blochsim import BlochSimulator, BlochVisualizer

# ---------- shared helper ----------
def plot_trajectory3d(sim, title=""):
    M = np.array([np.sum(s["magnetization"], axis=0) for s in sim.history])  # (T,3)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    # sphere surface (reference)
    u = np.linspace(0, 2*np.pi, 72)
    v = np.linspace(0, np.pi, 36)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.12, edgecolor="none")

    ax.plot(M[:, 0], M[:, 1], M[:, 2], linewidth=2)
    ax.scatter(M[0, 0], M[0, 1], M[0, 2], s=60, c="g", label="start")
    ax.scatter(M[-1, 0], M[-1, 1], M[-1, 2], s=60, c="r", label="end")
    ax.legend()
    ax.set_xlabel("Mx"); ax.set_ylabel("My"); ax.set_zlabel("Mz")
    lim = 1.2
    ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim]); ax.set_zlim([-lim, lim])
    ax.view_init(elev=25, azim=35)
    return fig

# ---------- each case returns a configured-and-run simulator ----------
def case_free_precession():
    """Single spin; precession around +z with no relaxation."""
    sim = BlochSimulator(gamma=1.0, B0=2.0)
    sim.setup_single_spin()
    sim.isochromats[0].M = np.array([1.0, 0.0, 0.0])  # start in xy-plane
    sim.simulate_sequence(duration=5.0, dt=0.001)
    return sim, "Free precession (B0 only)"

def case_90_then_free():
    """90° hard pulse tips M to xy; observe post-pulse precession."""
    sim = BlochSimulator(gamma=1.0, B0=2.0)
    sim.setup_single_spin()
    sim.RF_pulse_rect(angle=np.pi/2, phase=0.0, B1_amplitude=4.0)
    sim.simulate_sequence(duration=3.0, dt=0.0005)
    return sim, "90° pulse → free precession"

def case_relaxation_only():
    """No B0; watch pure T1 recovery and T2 decay toward equilibrium."""
    sim = BlochSimulator(gamma=1.0, B0=0.0)
    sim.isochromats.clear()
    sim.add_isochromat(M=np.array([1.0, 0.0, 0.0]), T1=7.0, T2=1.0)
    sim.simulate_sequence(duration=8.0, dt=0.005)
    return sim, "Relaxation only (T1/T2)"

def case_relaxation_plus_precession():
    """B0 ≠ 0; spiral back to +z while precessing."""
    sim = BlochSimulator(gamma=1.0, B0=2.0)
    sim.isochromats.clear()
    sim.add_isochromat(M=np.array([1.0, 0.0, 0.0]), T1=9.0, T2=2.0)
    sim.simulate_sequence(duration=10.0, dt=0.005)
    return sim, "Relaxation + precession"

def case_inhomogeneous_dephasing(n=15, spread=0.5):
    """Multiple spins with different dB0; transverse dephasing (T2*)."""
    sim = BlochSimulator(gamma=1.0, B0=2.0)
    sim.setup_inhomogeneous(n_spins=n, spread=spread)
    # tip all into xy to see dephasing clearly
    sim.RF_pulse_rect(angle=np.pi/2, phase=0.0, B1_amplitude=5.0)
    sim.simulate_sequence(duration=2.0, dt=0.0005)
    return sim, "Inhomogeneous dephasing (T2*)"

def case_gradient_dephase_rephase():
    """
    Simple 'gradient echo' style: turn on Gx briefly to dephase, then flip sign to rephase.
    NOTE: your apply_gradient sets a steady gradient; we emulate finite pulses by setting Gx,
    simulating, then zeroing / negating it.
    """
    sim = BlochSimulator(gamma=1.0, B0=0.0)
    sim.setup_spatial_line(n_spins=41, spacing=0.15)
    sim.RF_pulse_rect(angle=np.pi/2, phase=0.0, B1_amplitude=6.0)
    sim.simulate_sequence(duration=0.05, dt=0.0005)   # small delay

    # "Dephase" with +Gx for τ
    sim.Gx = 8.0; sim.Gy = 0.0
    sim.simulate_sequence(duration=0.05, dt=0.0005)

    # "Rephase" with -Gx for τ
    sim.Gx = -8.0
    sim.simulate_sequence(duration=0.05, dt=0.0005)

    # gradients off
    sim.Gx = 0.0
    sim.simulate_sequence(duration=0.05, dt=0.0005)
    return sim, "Gradient echo: dephase → rephase"

def case_spin_echo():
    """Classic Hahn spin echo: 90° — τ — 180° — τ; refocus inhomogeneous dephasing."""
    sim = BlochSimulator(gamma=1.0, B0=2.0)
    sim.setup_inhomogeneous(n_spins=21, spread=0.5)

    # 90° to transverse
    sim.RF_pulse_rect(angle=np.pi/2, phase=0.0, B1_amplitude=6.0)
    sim.simulate_sequence(duration=0.05, dt=0.0005)   # τ

    # 180° inversion about x̂ (phase=0)
    sim.RF_pulse_rect(angle=np.pi, phase=0.0, B1_amplitude=6.0)
    sim.simulate_sequence(duration=0.05, dt=0.0005)   # τ

    # observe echo
    sim.simulate_sequence(duration=0.05, dt=0.0005)
    return sim, "Hahn spin echo (90-τ-180-τ)"

def case_sinc_pulse_slice_like():
    """
    Sinc-shaped RF pulse to show controlled flip-area; optional weak gradient
    to mimic slice-select coupling qualitatively (in this 0D/1D model).
    """
    sim = BlochSimulator(gamma=1.0, B0=2.0)
    sim.setup_spatial_line(n_spins=41, spacing=0.1)
    # weak gradient on during RF (emulated by leaving Gx set)
    sim.Gx = 1.5
    sim.RF_pulse_sinc(angle=np.pi/2, phase=0.0, B1_amplitude=5.0, n_lobes=4)
    sim.simulate_sequence(duration=0.04, dt=0.0004)
    sim.Gx = 0.0
    sim.simulate_sequence(duration=0.06, dt=0.0004)
    return sim, "Sinc RF (slice-like demo)"

# ---------- registry ----------
CASES = {
    "free_precession": case_free_precession,
    "rf_90_then_free": case_90_then_free,
    "relaxation_only": case_relaxation_only,
    "relaxation_plus_precession": case_relaxation_plus_precession,
    "inhomogeneous_dephasing": case_inhomogeneous_dephasing,
    "gradient_echo": case_gradient_dephase_rephase,
    "spin_echo": case_spin_echo,
    "sinc_pulse": case_sinc_pulse_slice_like,
}

def run_case(name: str):
    if name not in CASES:
        raise ValueError(f"Unknown case '{name}'. Options: {list(CASES)}")
    sim, title = CASES[name]()
    # 2D traces
    fig2d = BlochVisualizer.plot_magnetization_trajectory(sim)
    fig2d.suptitle(title)
    plt.show()

    # Snapshot on Bloch sphere
    BlochVisualizer.plot_bloch_sphere(sim, frame_idx=-1)
    plt.title(f"{title} — final snapshot")
    plt.show()

    # 3D path of total M
    plot_trajectory3d(sim, f"{title} — 3D trajectory")
    plt.show()
    return sim

if __name__ == "__main__":
    run_case("relaxation_plus_precession")
