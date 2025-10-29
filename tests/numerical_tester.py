import experiment_numerical as exp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Physical parameters
gamma = 2 * np.pi * 42.58e6   # rad/s/T for proton
B = np.array([0, 0, 0.01])       # 1 Tesla field along z
T1 = 1.5                      # s
T2 = 0.1                      # s
M0 = 1.0                      # equilibrium Mz
M_init = np.array([1.0, 0.0, 0.0])  # start along x

# Simulation parameters
t_max = 1e-4       # 0.1 ms
dt = 1e-8          # 10 ns step

# running
t_eu, M_eu = exp.simulate_bloch(exp.euler_step, M_init, t_max, dt, gamma, T1, T2, M0, B)
t_rk, M_rk = exp.simulate_bloch(exp.rk4_step, M_init, t_max, dt, gamma, T1, T2, M0, B)


# plotting x, y, z vs time
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

labels = ['Mx', 'My', 'Mz']
colors = ['r', 'b', 'g']

for i, ax in enumerate(axes):
    ax.plot(t_eu * 1e3, M_eu[:, i], 'r' + '--', label=f'{labels[i]} (Euler)')
    ax.plot(t_rk * 1e3, M_rk[:, i], 'b', label=f'{labels[i]} (RK4)')
    ax.set_ylabel(f'{labels[i]}')
    ax.legend()
    ax.grid(True)

axes[-1].set_xlabel('Time (ms)')
fig.suptitle('Bloch Equation Components vs Time (Euler vs RK4)', fontsize=12)
plt.tight_layout()
plt.show()

# 3d plot
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(M_rk[:, 0], M_rk[:, 1], M_rk[:, 2], color='purple')
ax.set_xlabel('Mx')
ax.set_ylabel('My')
ax.set_zlabel('Mz')
ax.set_title('3D Bloch Vector Trajectory (RK4)')
plt.show()