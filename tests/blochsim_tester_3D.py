import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from blochsim import BlochSimulator, BlochVisualizer


# --- helper: draw 3D path ---
def plot_trajectory3d(sim, title=""):
    M = np.array([np.sum(s["magnetization"], axis=0) for s in sim.history])  # (T,3)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    # sphere surface for reference
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

    ax.set_xlabel("Mx")
    ax.set_ylabel("My")
    ax.set_zlabel("Mz")
    lim = 1.2
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.view_init(elev=25, azim=35)
    return fig


# --- TEST 1: Free precession under B0 only ---
def test_free_precession_3d():
    sim = BlochSimulator(gamma=1.0, B0=2.0)
    sim.setup_single_spin()
    sim.isochromats[0].M = np.array([1.0, 0.0, 0.0])  # start in xy plane
    sim.simulate_sequence(duration=10.0, dt=0.001)  # ⬆️ duration

    fig2d = BlochVisualizer.plot_magnetization_trajectory(sim)
    fig2d.suptitle("Test 1: Free precession (10 s)")
    plt.show()

    BlochVisualizer.plot_bloch_sphere(sim, frame_idx=-1)
    plt.title("Test 1: Final snapshot on Bloch sphere")
    plt.show()

    plot_trajectory3d(sim, "Test 1: 3D trajectory (10 s of precession)")
    plt.show()


# --- TEST 2: 90° RF pulse then precession ---
def test_rf_pulse_3d():
    sim = BlochSimulator(gamma=1.0, B0=2.0)
    sim.setup_single_spin()
    sim.RF_pulse_rect(angle=np.pi / 2, phase=0.0, B1_amplitude=4.0)
    # run longer to watch post-pulse precession
    sim.simulate_sequence(duration=5.0, dt=0.0005)

    fig2d = BlochVisualizer.plot_magnetization_trajectory(sim)
    fig2d.suptitle("Test 2: 90° RF pulse → tip into XY plane (5 s run)")
    plt.show()

    BlochVisualizer.plot_bloch_sphere(sim, frame_idx=-1)
    plt.title("Test 2: Final snapshot on Bloch sphere")
    plt.show()

    plot_trajectory3d(sim, "Test 2: 3D trajectory (5 s post-pulse precession)")
    plt.show()


# --- TEST 3: Relaxation (finite T1, T2) ---
def test_relaxation_3d():
    sim = BlochSimulator(gamma=1.0, B0=0.0)
    sim.isochromats.clear()
    sim.add_isochromat(M=np.array([1.0, 0.0, 0.0]), T1=3.0, T2=1.5)
    sim.simulate_sequence(duration=8.0, dt=0.005)  # ⬆️ longer duration

    fig2d = BlochVisualizer.plot_magnetization_trajectory(sim)
    fig2d.suptitle("Test 3: Relaxation (T₁=3 s, T₂=1.5 s, 8 s total)")
    plt.show()

    BlochVisualizer.plot_bloch_sphere(sim, frame_idx=-1)
    plt.title("Test 3: Final snapshot on Bloch sphere")
    plt.show()

    plot_trajectory3d(sim, "Test 3: 3D trajectory (relaxation to equilibrium)")
    plt.show()

def test_relaxation_with_precession_3d():
    sim = BlochSimulator(gamma=1.0, B0=2.0)   # <-- nonzero B0 -> precession about +z
    sim.isochromats.clear()
    sim.add_isochromat(M=np.array([1.0, 0.0, 0.0]), T1=3.0, T2=1.5)  # start transverse
    sim.simulate_sequence(duration=8.0, dt=0.005)

    # 2D traces
    fig2d = BlochVisualizer.plot_magnetization_trajectory(sim)
    fig2d.suptitle("Relaxation + Precession (T₁=3s, T₂=1.5s, B₀=2)")
    plt.show()

    # 3D snapshot + full 3D path
    BlochVisualizer.plot_bloch_sphere(sim, frame_idx=-1)
    plt.title("Final snapshot on Bloch sphere (Relaxation + Precession)")
    plt.show()

    plot_trajectory3d(sim, "3D trajectory: spiral toward +Z (precession + T₁/T₂)")
    plt.show()


if __name__ == "__main__":
    print("Running extended 3D visual tests…")
    test_free_precession_3d()
    test_rf_pulse_3d()
    test_relaxation_3d()
    test_relaxation_with_precession_3d()
