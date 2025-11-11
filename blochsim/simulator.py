import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import json


@dataclass # class that just holds data
class Isochromat: # spin packet: group of spins that are @ same loc and experience same B field.
    """Represents a single isochromat (spin packet) with position and magnetization"""
    M: np.ndarray  # Magnetization vector [Mx, My, Mz] (current magnetization vector of packet)
    pos: np.ndarray  # Position [x, y, z] (spatial position of packet, used in gradient sim.)
    dB0: float = 0.0  # Field offset (field inhomogeneity. irl, B0 isn't uniform & dB0 is small constant 
                      # offset. primary cause of signal decay.)
    T1: float = np.inf  # Longitudinal relaxation time (how quickly component along main field M_z recovers to o.g. M_0
                        # small T_1 => fast recovery)
    T2: float = np.inf  # Transverse relaxation time (how quickly M_x and M_y decay to 0 b/c of random microscopic spin
                        # interactions.)
    M0: float = 1.0  # Equilibrium magnetization (fully relaxed magnetization vector in +z axis. )


class BlochSimulator:
    
    def __init__(self, gamma: float = 1.0, B0: float = 2.0):
        """
            gamma: Gyromagnetic ratio (default normalized to 1) (omega = gamma * B)
            B0: Main magnetic field strength
        """
        self.gamma = gamma 
        self.B0 = B0 # strength of main static B field
        self.isochromats: List[Isochromat] = [] # list of all spin packets
        self.time = 0.0 # current time ig
        self.history = [] # snapshot of system state at every time step
        
        # Field parameters
        self.B1 = 0.0  # RF field amplitude
        self.B1_freq = 0.0  # RF frequency
        self.phi1 = 0.0  # RF phase
        self.Gx = 0.0  # X gradient
        self.Gy = 0.0  # Y gradient
        
        # Pulse parameters
        self.RF_active = False
        self.RF_time_left = 0.0
        self.RF_area_left = 0.0
        self.RF_func: Optional[Callable] = None
        
    def add_isochromat(self, M: np.ndarray, pos: np.ndarray = np.array([0, 0, 0]),
                       dB0: float = 0.0, T1: float = np.inf, T2: float = np.inf):
        # Add an isochromat to the simulation
        iso = Isochromat(M=M.copy(), pos=pos.copy(), dB0=dB0, T1=T1, T2=T2)
        self.isochromats.append(iso)
        
    def setup_single_spin(self):
        # Setup a single perfect spin at origin w/o relaxation equilibrium 
        self.isochromats.clear()
        self.add_isochromat(np.array([0, 0, 1]))
        
    def setup_inhomogeneous(self, n_spins: int = 9, spread: float = 0.3):
        # Setup multiple spins with field inhomogeneity. Multiple spins @ same location but diff. dBO values. Simulate T2 decay
        # where diff spins precess @ diff speeds and fan out in xy plane (causing signal decay).
        self.isochromats.clear()
        for i in range(n_spins):
            dB0 = (i - (n_spins-1)/2) / (n_spins/6) * spread
            self.add_isochromat(np.array([0, 0, 1]), dB0=dB0)
            
    def setup_spatial_line(self, n_spins: int = 21, spacing: float = 0.4):
        # Setup spins along a line (x-axis) for gradient testing. Show if G_x (x gradient) works.
        self.isochromats.clear()
        for i in range(n_spins):
            x_pos = (i - (n_spins-1)/2) * spacing
            self.add_isochromat(
                np.array([0, 0, 1]), 
                pos=np.array([x_pos, 0, 0])
            )
            
    def setup_relaxation(self, T1: float = 5.0, T2: float = 3.0):
        # Setup a single spin with relaxation (finite T1, T2 values). 
        self.isochromats.clear()
        self.add_isochromat(np.array([0, 0, 1]), T1=T1, T2=T2)
        
    def RF_pulse_rect(self, angle: float, phase: float, B1_amplitude: float = 4.0):
        """
        Apply rectangular RF pulse

            angle: Flip angle in radians
            phase: RF phase in radians
            B1_amplitude: RF field strength
        """
        self.B1 = B1_amplitude
        self.B1_freq = self.gamma * self.B0
        self.phi1 = phase
        duration = angle / (self.gamma * B1_amplitude)
        
        self.RF_active = True
        self.RF_time_left = duration
        self.RF_area_left = angle
        self.RF_func = self._RF_const # which function step method shoudl call to get B1 field value @ some time.
        
    def RF_pulse_sinc(self, angle: float, phase: float, B1_amplitude: float = 4.0,
                      n_lobes: int = 4):
        # a sinc-shaped RF pulse
        self.B1 = B1_amplitude
        self.B1_freq = self.gamma * self.B0
        self.phi1 = phase
        
        # Sinc pulse is longer to maintain same area
        sinc_correction = 0.22570583339507  # Si(2π)/(2π) for 4-lobe sinc
        duration = angle / (self.gamma * B1_amplitude * sinc_correction)
        
        self.RF_active = True
        self.RF_time_left = duration
        self.RF_area_left = angle
        self.RF_func = lambda t: self._RF_sinc(t, duration, n_lobes)

    # RF_const and RF_sinc actually calculate B1 vector at time t.  
    def _RF_const(self, t: float) -> Tuple[np.ndarray, float]:
        # Constant (rectangular) RF pulse shape
        phase = self.B1_freq * t - self.phi1
        B1_vec = np.array([
            self.B1 * np.cos(phase),
            -self.B1 * np.sin(phase),
            0.0
        ])
        return B1_vec, self.B1
    
    def _RF_sinc(self, t: float, duration: float, n_lobes: int) -> Tuple[np.ndarray, float]:
        # Sinc RF pulse shape
        phase = self.B1_freq * t - self.phi1
        sinc_arg = n_lobes * np.pi * (t/duration - 0.5)
        
        if abs(sinc_arg) > 0.01:
            envelope = self.B1 * np.sin(sinc_arg) / sinc_arg
        else:
            envelope = self.B1
            
        B1_vec = np.array([
            envelope * np.cos(phase),
            -envelope * np.sin(phase),
            0.0
        ])
        return B1_vec, envelope
    
    import numpy as np

    # ----------------------- Helpers (inside the class) -----------------------

    def _duration_from_envelope_avg(self, angle, B1_peak, env_of_tau, n=4096):
        """
        Numerically compute the duration that yields the requested flip angle.

        We assume an envelope defined on tau in [0,1] with peak 1.0:
            B1(t) = B1_peak * env_of_tau(t / duration)
        => area = ∫ B1 dt = B1_peak * duration * ∫_0^1 env(τ) dτ
        Flip angle: angle = gamma * area

        So: duration = angle / (gamma * B1_peak * avg_env),
            where avg_env = ∫_0^1 env(τ) dτ
        """
        tau = np.linspace(0.0, 1.0, n, endpoint=False) + 0.5 / n
        env_vals = env_of_tau(tau)
        avg_env = np.trapz(env_vals, tau)  # integral over [0,1]
        avg_env = float(max(avg_env, 1e-12))  # guard
        return angle / (self.gamma * B1_peak * avg_env)

    # -------------------- 1) Gaussian pulse (RF_pulse_gaussian) --------------------

    def RF_pulse_gaussian(self, angle: float, phase: float, B1_amplitude: float = 4.0,
                        sigma_frac: float = 0.20):
        """
        Apply a Gaussian-envelope RF pulse with flip-angle normalization.

        angle: flip angle [rad]
        phase: RF carrier phase [rad]
        B1_amplitude: peak B1 (envelope peak = 1.0 -> peak field = B1_amplitude)
        sigma_frac: controls width of the Gaussian relative to duration
                    (σ in units of half-duration; 0.2 is a common smooth choice)
        """
        self.B1 = B1_amplitude
        self.B1_freq = self.gamma * self.B0
        self.phi1 = phase

        # envelope on τ ∈ [0,1], centered at 0.5
        def env_of_tau(tau):
            # convert sigma_frac (fraction of half-duration) to σ on τ-domain
            # half-duration in τ is 0.5, so σ_τ = sigma_frac / 0.5
            sigma_tau = sigma_frac / 0.5
            return np.exp(-0.5 * ((tau - 0.5) / sigma_tau) ** 2)

        duration = self._duration_from_envelope_avg(angle, B1_amplitude, env_of_tau)

        self.RF_active = True
        self.RF_time_left = duration
        self.RF_area_left = angle
        self.RF_func = lambda t: self._RF_gaussian(t, duration, sigma_frac)

    def _RF_gaussian(self, t: float, duration: float, sigma_frac: float):
        """Return (B1_vec, envelope) for Gaussian envelope at time t."""
        tau = np.clip(t / duration, 0.0, 1.0)
        sigma_tau = sigma_frac / 0.5
        env = float(np.exp(-0.5 * ((tau - 0.5) / sigma_tau) ** 2))

        phase = self.B1_freq * t - self.phi1
        B1x = self.B1 * env * np.cos(phase)
        B1y = -self.B1 * env * np.sin(phase)
        B1_vec = np.array([B1x, B1y, 0.0], dtype=float)
        return B1_vec, self.B1 * env

    # --------------- 2) Hamming-windowed sinc (RF_pulse_windowed_sinc) ---------------

    def RF_pulse_windowed_sinc(self, angle: float, phase: float, B1_amplitude: float = 4.0,
                            n_lobes: int = 4):
        """
        Apply a Hamming-windowed sinc RF pulse (cleaner spectrum than plain sinc).

        angle: flip angle [rad]
        phase: RF carrier phase [rad]
        B1_amplitude: peak B1 (after windowing; peak env is normalized to 1)
        n_lobes: number of lobes on each side of the center
        """
        self.B1 = B1_amplitude
        self.B1_freq = self.gamma * self.B0
        self.phi1 = phase

        # Continuous-time Hamming window on τ ∈ [0,1]
        def window_hamming(tau):
            return 0.54 - 0.46 * np.cos(2 * np.pi * tau)

        def env_of_tau(tau):
            # Map τ∈[0,1] to x∈[-n_lobes, n_lobes]
            x = (tau - 0.5) * 2.0 * n_lobes
            sinc = np.sinc(x)  # np.sinc uses sin(πx)/(πx); OK up to scale
            env = sinc * window_hamming(tau)
            # normalize to peak 1.0 so B1_amplitude is actual peak
            peak = np.max(np.abs(env))
            return env / (peak + 1e-12)

        duration = self._duration_from_envelope_avg(angle, B1_amplitude, env_of_tau)

        self.RF_active = True
        self.RF_time_left = duration
        self.RF_area_left = angle
        self.RF_func = lambda t: self._RF_windowed_sinc(t, duration, n_lobes)

    def _RF_windowed_sinc(self, t: float, duration: float, n_lobes: int):
        """Return (B1_vec, envelope) for Hamming-windowed sinc at time t."""
        tau = np.clip(t / duration, 0.0, 1.0)
        x = (tau - 0.5) * 2.0 * n_lobes
        sinc = np.sinc(x)
        window = 0.54 - 0.46 * np.cos(2 * np.pi * tau)
        env = sinc * window
        env /= (np.max([abs(env), 1e-12]))  # tiny guard if called at boundaries

        phase = self.B1_freq * t - self.phi1
        B1x = self.B1 * env * np.cos(phase)
        B1y = -self.B1 * env * np.sin(phase)
        B1_vec = np.array([B1x, B1y, 0.0], dtype=float)
        return B1_vec, self.B1 * env

    # ------------------ 3) Trapezoidal (ramps) (RF_pulse_trapezoid) ------------------

    def RF_pulse_trapezoid(self, angle: float, phase: float, B1_amplitude: float = 4.0,
                        ramp_frac: float = 0.10):
        """
        Apply a trapezoidal RF pulse with linear rise/fall ramps.

        angle: flip angle [rad]
        phase: RF carrier phase [rad]
        B1_amplitude: plateau peak
        ramp_frac: fraction of the duration used for each ramp (0–0.5)
        """
        ramp_frac = float(np.clip(ramp_frac, 0.0, 0.49))
        self.B1 = B1_amplitude
        self.B1_freq = self.gamma * self.B0
        self.phi1 = phase

        def env_of_tau(tau):
            # piecewise linear: rise -> flat -> fall on τ∈[0,1]
            r = ramp_frac
            env = np.empty_like(tau)
            # rise
            rise = tau < r
            env[rise] = tau[rise] / max(r, 1e-12)
            # flat
            flat = (tau >= r) & (tau <= 1.0 - r)
            env[flat] = 1.0
            # fall
            fall = tau > 1.0 - r
            env[fall] = (1.0 - tau[fall]) / max(r, 1e-12)
            return env

        duration = self._duration_from_envelope_avg(angle, B1_amplitude, env_of_tau)

        self.RF_active = True
        self.RF_time_left = duration
        self.RF_area_left = angle
        self.RF_func = lambda t: self._RF_trapezoid(t, duration, ramp_frac)

    def _RF_trapezoid(self, t: float, duration: float, ramp_frac: float):
        """Return (B1_vec, envelope) for a trapezoid at time t."""
        tau = np.clip(t / duration, 0.0, 1.0)
        r = ramp_frac
        if tau < r:
            env = tau / max(r, 1e-12)
        elif tau <= 1.0 - r:
            env = 1.0
        else:
            env = (1.0 - tau) / max(r, 1e-12)

        phase = self.B1_freq * t - self.phi1
        B1x = self.B1 * env * np.cos(phase)
        B1y = -self.B1 * env * np.sin(phase)
        B1_vec = np.array([B1x, B1y, 0.0], dtype=float)
        return B1_vec, self.B1 * env



    def random_fourier_pulse(T: int, dt: float, A: float, n_terms: int = 5):
        """
        Generate a random composite pulse as a sum of sinusoids.

        Parameters
        ----------
        T : int
            Number of samples.
        dt : float
            Time step (seconds).
        A : float
            Overall amplitude scaling.
        n_terms : int
            Number of random harmonic components to sum.

        Returns
        -------
        np.ndarray
            Random, smooth RF envelope normalized to amplitude A.
        """
        t = np.arange(T) * dt
        y = np.zeros_like(t)
        for _ in range(n_terms):
            f = np.random.uniform(100, 5000)  # random frequency in Hz
            phi = np.random.uniform(0, 2 * np.pi)
            y += np.sin(2 * np.pi * f * t + phi)
        y /= np.max(np.abs(y)) + 1e-8
        return (A * y).astype(np.float32)



    def apply_gradient(self, phase_diff: float, direction_angle: float = 0.0):
        """Apply gradient pulse"""
        grad_scale = 11  # From o.g. code
        duration = 1.0  # seconds
        area = phase_diff * grad_scale / self.gamma
        
        self.Gx = np.cos(direction_angle) * area / duration
        self.Gy = np.sin(direction_angle) * area / duration
        
    def step(self, dt: float):
        """
        Evolve the system by one time step using Bloch equations

            dt: Time step in seconds
        """
        self.time += dt
        
        # Get RF field
        if self.RF_active and self.RF_time_left > 0: # check if RF is on and time remaining
            B1_vec, envelope = self.RF_func(self.time) # get B1 vector
            dArea = dt * self.gamma * envelope
            
            if self.RF_time_left < dt:
                # End of pulse (adjust to match target angle)
                B1_vec *= self.RF_area_left / dArea
                self.RF_active = False
                self.RF_time_left = 0
                self.B1 = 0
            else:
                self.RF_area_left -= dArea
                self.RF_time_left -= dt
        else: # no B1 vec if no RF
            B1_vec = np.array([0, 0, 0])
            self.RF_active = False
            
        # Update each isochromat
        for iso in self.isochromats:
            # Total field including gradients and offsets. total B field each iso feels at this instant.
            grad_scale = 11  # From o.g. code
            detuning = iso.dB0 + (self.Gx * iso.pos[0] + self.Gy * iso.pos[1]) / grad_scale
            # z field detuning is sum of spin's personal offset and gradient fields.  multiplications mean "if G_x grad on, 
            # field depends on x_position)"
            B_total = B1_vec + np.array([0, 0, self.B0 + detuning])
            
            # Precession (M x B term) (Rotation due to B field)
            B_mag = np.linalg.norm(B_total)
            if B_mag > 1e-10:
                rotation_angle = -B_mag * dt * self.gamma
                rotation_axis = B_total / B_mag
                
                # Rodrigues' rotation formula
                K = np.array([
                    [0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0]
                ])
                R = np.eye(3) + np.sin(rotation_angle) * K + \
                    (1 - np.cos(rotation_angle)) * K @ K
                iso.M = R @ iso.M # isochromat's magnetizatoin vector updated by matrix-multiplying it by rotation
                # matrix R to precess vector.
            
            # Relaxation (T1, T2 terms)
            if np.isfinite(iso.T1) or np.isfinite(iso.T2):
                f1 = np.exp(-dt / iso.T1) if np.isfinite(iso.T1) else 1.0 # exponential decay factor for M_z in time dt
                f2 = np.exp(-dt / iso.T2) if np.isfinite(iso.T2) else 1.0 # exponential decay factor for M_x and M_y in time dt.
                
                iso.M[0] *= f2 # Mx decays toward 0
                iso.M[1] *= f2 # My decays toward 0
                iso.M[2] = iso.M[2] * f1 + (1 - f1) * iso.M0 # analytic solution for T1 relaxation. M_z decays toward 0 w/
                # f1 and (at teh same time) grows back towards M0 (equil)
        
        # Store history
        self._record_state() # append state of all isochromats to self.history
        
    def _record_state(self):
        """Record current state for history"""
        state = {
            'time': self.time,
            'magnetization': np.array([iso.M.copy() for iso in self.isochromats]),
            'B1': self.B1,
            'RF_active': self.RF_active
        }
        self.history.append(state)
        
    def get_total_magnetization(self) -> np.ndarray:
        # Get total magnetization vector
        return np.sum([iso.M for iso in self.isochromats], axis=0)
    
    def get_transverse_magnetization(self) -> float:
        # Get magnitude of transverse magnetization
        M_total = self.get_total_magnetization()
        return np.sqrt(M_total[0]**2 + M_total[1]**2)
    
    def simulate_sequence(self, duration: float, dt: float = 0.01):
        """
        Run simulation for specified duration

            duration: Total simulation time
            dt: Time step
        """
        steps = int(duration / dt)
        for _ in range(steps):
            self.step(dt)
            
    def reset(self):
        # Reset simulation
        self.time = 0.0
        self.history.clear()
        for iso in self.isochromats:
            iso.M = np.array([0, 0, iso.M0])


class BlochVisualizer:
    # Visualization for Bloch simulation
    
    @staticmethod
    def plot_magnetization_trajectory(simulator: BlochSimulator, figsize=(12, 4)):
        # Plot Mx, My, Mz over time. Calculates total magnetizaiotn by summing M vectors of all isochromats
        # at each time step.
        if not simulator.history:
            print("No simulation data to plot")
            return
            
        times = [state['time'] for state in simulator.history]
        M_total = np.array([
            np.sum(state['magnetization'], axis=0) / len(simulator.isochromats)
            for state in simulator.history
        ])
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        axes[0].plot(times, M_total[:, 0], 'r-', linewidth=2)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Mx')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('X Magnetization')
        
        axes[1].plot(times, M_total[:, 1], 'g-', linewidth=2)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('My')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Y Magnetization')
        
        axes[2].plot(times, M_total[:, 2], 'b-', linewidth=2)
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Mz')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title('Z Magnetization')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_bloch_sphere(simulator: BlochSimulator, frame_idx: int = -1):
        # Plot magnetization vectors on Bloch sphere. Default picks last time step. Draw M vector for each isochromat.
        # use for dephasing visual. (run setup_inhomo and vectors start aligned and then fan out across xy plane)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, alpha=0.1, color='lightblue')
        
        # Draw axes
        ax.plot([0, 1.3], [0, 0], [0, 0], 'k-', linewidth=2, alpha=0.3)
        ax.plot([0, 0], [0, 1.3], [0, 0], 'k-', linewidth=2, alpha=0.3)
        ax.plot([0, 0], [0, 0], [0, 1.3], 'k-', linewidth=2, alpha=0.3)
        ax.text(1.4, 0, 0, 'X', fontsize=14)
        ax.text(0, 1.4, 0, 'Y', fontsize=14)
        ax.text(0, 0, 1.4, 'Z', fontsize=14)
        
        # Plot magnetization vectors
        if simulator.history:
            state = simulator.history[frame_idx]
            for M in state['magnetization']:
                ax.quiver(0, 0, 0, M[0], M[1], M[2],
                         arrow_length_ratio=0.1, linewidth=2, color='red')
        
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.set_xlabel('Mx')
        ax.set_ylabel('My')
        ax.set_zlabel('Mz')
        ax.set_title(f'Bloch Sphere at t={simulator.history[frame_idx]["time"]:.2f}s')
        
        return fig
    
    @staticmethod
    def plot_fid_spectrum(simulator: BlochSimulator):
        # Plot FID signal and its spectrum. simulates what MRI/NMR scanner actually measures.
        if not simulator.history:
            print("No simulation data to plot")
            return
            
        times = np.array([state['time'] for state in simulator.history])
        M_total = np.array([
            np.sum(state['magnetization'], axis=0)
            for state in simulator.history
        ])
        
        # Complex transverse magnetization (FID)
        signal = M_total[:, 0] + 1j * M_total[:, 1] # scanner's reciever coil picks up oscillating transverse xy magnetization
        # represnted as complex number. time domain signal called free induction decay.
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Time domain (real, im, and mag of signal that decays)
        axes[0].plot(times, signal.real, 'b-', label='Real', linewidth=2)
        axes[0].plot(times, signal.imag, 'r-', label='Imaginary', linewidth=2)
        axes[0].plot(times, np.abs(signal), 'k--', label='Magnitude', linewidth=2)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Signal')
        axes[0].set_title('FID Signal')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Frequency domain (spectrum)
        dt = times[1] - times[0] if len(times) > 1 else 0.01
        spectrum = np.fft.fftshift(np.fft.fft(signal)) # calculates FFT of FID. time to frequency domain.
        freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), dt)) # 0 freq at center
        
        axes[1].plot(freqs, np.abs(spectrum), 'b-', linewidth=2)
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Magnitude')
        axes[1].set_title('Spectrum')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([freqs[0], freqs[-1]])
        
        plt.tight_layout()
        return fig