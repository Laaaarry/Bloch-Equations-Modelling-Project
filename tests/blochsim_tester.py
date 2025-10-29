import numpy as np
import matplotlib.pyplot as plt
from blochsim import BlochSimulator, BlochVisualizer


# ---------- TEST 1: Free precession under B0 only ----------
def test_free_precession():
    sim = BlochSimulator(gamma=1.0, B0=2.0)
    sim.setup_single_spin()  # Start with M = [0, 0, 1]
    # Tip spin into XY plane manually
    sim.isochromats[0].M = np.array([1.0, 0.0, 0.0])
    sim.simulate_sequence(duration=2.0, dt=0.001)
    fig = BlochVisualizer.plot_magnetization_trajectory(sim)
    fig.suptitle("Test 1: Free precession (B₀ only)")
    plt.show()


# ---------- TEST 2: 90° RF pulse ----------
def test_rf_pulse():
    sim = BlochSimulator(gamma=1.0, B0=2.0)
    sim.setup_single_spin()
    sim.RF_pulse_rect(angle=np.pi / 2, phase=0.0, B1_amplitude=4.0)
    sim.simulate_sequence(duration=1.0, dt=0.0005)
    fig = BlochVisualizer.plot_magnetization_trajectory(sim)
    fig.suptitle("Test 2: 90° RF pulse (tip from +Z into XY plane)")
    plt.show()


# ---------- TEST 3: Relaxation (finite T1, T2) ----------
def test_relaxation():
    sim = BlochSimulator(gamma=1.0, B0=0.0)
    sim.isochromats.clear()
    sim.add_isochromat(M=np.array([1.0, 0.0, 0.0]), T1=2.0, T2=1.0)
    sim.simulate_sequence(duration=5.0, dt=0.005)
    fig = BlochVisualizer.plot_magnetization_trajectory(sim)
    fig.suptitle("Test 3: Relaxation (T₁ = 2s, T₂ = 1s)")
    plt.show()


if __name__ == "__main__":
    print("Running BlochSimulator visual tests...")
    test_free_precession()
    test_rf_pulse()
    test_relaxation()
