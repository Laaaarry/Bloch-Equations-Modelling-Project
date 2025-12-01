import experiment_numerical as exp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm.auto import tqdm

def analytical_soln_simple(M_init, t, gamma, T1, T2, M0, B0, dB0):
    """
    Analytic Bloch solution for simple case:
      - no RF or gradients (u = 0)
      - constant B0 along z
      - relaxation with T1, T2
      - precession around z: omega = gamma * (B0 + dB0)

    """
    Mx0, My0, Mz0 = M_init

    Mxy0 = np.sqrt(Mx0**2 + My0**2 + 1e-12)
    phi0 = np.arctan2(My0, Mx0)

    E1 = np.exp(-t / (T1 + 1e-12))
    E2 = np.exp(-t / (T2 + 1e-12))

    omega = -gamma * (B0 + dB0)
    phase = omega * t + phi0

    Mx = Mxy0 * E2 * np.cos(phase)
    My = Mxy0 * E2 * np.sin(phase)
    Mz = M0 + (Mz0 - M0) * E1

    M = np.column_stack([Mx, My, Mz])
    return M

def gen_test_case(test_id = 0):

    np.random.seed(123 + test_id)

    # physical parameters
    M_init = np.random.randn(3)
    M_init = M_init / (np.linalg.norm(M_init) + 1e-12)
    t_max = 10
    dt = 0.1
    gamma = 1 # rad/s/T for proton
    T1 = np.random.uniform(2, 5.0)
    T2 = np.random.uniform(0.6, 2.0)
    M0 = 1.0 
    B0  = np.random.uniform(2.0, 4.0)
    dB0 = np.random.uniform(-0.1, 0.1)

    return (M_init, t_max, dt, gamma, T1, T2, M0, B0, dB0)

def compare(test_id=0):

    M_init, t_max, dt, gamma, T1, T2, M0, B0, dB0 = gen_test_case(test_id)
    print(f'================TEST CASE {test_id}================')
    print(f'Parameters')
    print(f'Initial Magnetization: {M_init}')
    print(f'Simulation Time: {t_max} ms')
    print(f'Time step: {dt}')
    print(f'Gyromagnetic Ratio: {gamma}')
    print(f'T1 Relaxation: {T1}')
    print(f'T2 Relaxation: {T2}')
    print(f'Equilibrium Magnetization: {M0}')
    print(f'Static Field: {B0}')
    print(f'B0 Inhomogeneity: {dB0}')
    print('____________________________________________________')

    B_field = np.array([0.0, 0.0, float(B0) + float(dB0)])

    print('Computing numerical solutions...')
    t_eu, M_eu = exp.simulate_bloch(exp.euler_step, M_init, t_max, dt, gamma, T1, T2, M0, B_field)
    t_rk, M_rk = exp.simulate_bloch(exp.rk4_step, M_init, t_max, dt, gamma, T1, T2, M0, B_field)
    t_rkf, M_rkf = exp.bloch_rkf(exp.rkf_step, M_init, t_max, dt, gamma, T1, T2, M0, B_field)

    # print(M_eu.shape) #(10001, 3)
    # print(M_rk.shape) #(10001, 3)

    print(f"Euler: {len(t_eu)} points")
    print(f"RK4:   {len(t_rk)} points")
    print(f"RKF:   {len(t_rkf)} points (adaptive)")
   
    print('Computing analytical solutions...')
    M_analytical_eu = analytical_soln_simple(M_init, t_eu, gamma, T1, T2, M0, B0, dB0)
    M_analytical_rk4 = analytical_soln_simple(M_init, t_rk, gamma, T1, T2, M0, B0, dB0)
    M_analytical_rkf = analytical_soln_simple(M_init, t_rkf, gamma, T1, T2, M0, B0, dB0)
    
   
    res = {
        't_eu': t_eu, 'M_eu': M_eu,
        't_rk': t_rk, 'M_rk': M_rk,
        't_rkf': t_rkf, 'M_rkf': M_rkf,
        'M_analytic_eu': M_analytical_eu,
        'M_analytic_rk': M_analytical_rk4,
        'M_analytic_rkf': M_analytical_rkf
    }

    visual_solution(res)
    visual_error(res)


    return res

def visual_solution(res):
    # unpacking results
    t_eu = res['t_eu']
    M_eu = res['M_eu']
    t_rk = res['t_rk']
    M_rk = res['M_rk']
    t_rkf = res['t_rkf']
    M_rkf = res['M_rkf']
    M_analytical_eu = res['M_analytic_eu']
    M_analytical_rk4 = res['M_analytic_rk']
    M_analytical_rkf = res['M_analytic_rkf']

    # convert to milliseconds
    t_eu_ms = t_eu * 1e3
    t_rk_ms = t_rk * 1e3
    t_rkf_ms = t_rkf * 1e3

    methods = [
    ('Euler', t_eu_ms, M_eu, M_analytical_eu),
    ('RK4', t_rk_ms, M_rk,M_analytical_rk4), 
    ('RKF', t_rkf_ms, M_rkf,M_analytical_rkf)
    ]
    
    # figure 1: separate graphs with 3d trajectories 
    fig1, axes1 = plt.subplots(2, 3, figsize = (15,8))
    components = [('Mx', 'r-'), ('My', 'b-'), ('Mz', 'g-')]
    for col, (method_name, t_ms, M, M_analytical) in enumerate(methods):

        ax = axes1[0, col]

        # time graph
        for row, (comp_name, linestyle) in enumerate (components):  # Mx, My, Mz
    
            ax.plot(t_ms, M[:, row], linestyle, label=comp_name, alpha=0.8)
            ax.plot(t_ms, M_analytical[:, row], color='k', linestyle='--')
       
        ax.set_ylabel('Magnetization')
        ax.set_title(f'{method_name}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # 3d trajectory
        ax3d = axes1[1, col]
        ax3d.remove()
        ax3d = fig1.add_subplot(2, 3, col + 4, projection='3d')
        ax3d.plot(M[:, 0], M[:, 1], M[:, 2], color=linestyle[0], alpha=0.8)
        ax3d.scatter(M[0, 0], M[0, 1], M[0, 2], 
                    color='k', s=50, label='Start')
        ax3d.scatter(M[-1, 0], M[-1, 1], M[-1, 2],
                    color='k', s=50, label='End')
        ax3d.set_xlabel('Mx')
        ax3d.set_ylabel('My')
        ax3d.set_zlabel('Mz')
        ax3d.set_title(f'{method_name} Trajectory')
        ax3d.legend()

    # figure 2: overlayed time graphs
    # plotting x, y, z vs time
    fig2, axes2 = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    labels = ['Mx', 'My', 'Mz']

    for i, ax in enumerate(axes2):
        ax.plot(t_eu * 1e3, M_eu[:, i], 'r', label=f'{labels[i]} (Euler)')
        ax.plot(t_rk * 1e3, M_rk[:, i], 'b', label=f'{labels[i]} (RK4)')
        ax.plot(t_rkf * 1e3, M_rkf[:, i], 'g', label=f'{labels[i]} (RKF)')
        ax.set_ylabel(f'{labels[i]}')
        ax.legend()
        ax.grid(True)
    axes2[-1].set_xlabel('Time (ms)')
    fig2.suptitle('Bloch Equation Components vs Time', fontsize=12)
    plt.tight_layout()
    plt.show()

def visual_error(res):
    # unpacking results
    t_eu = res['t_eu'] * 1e3
    t_rk = res['t_rk'] * 1e3
    t_rkf = res['t_rkf'] * 1e3
    M_eu = res['M_eu']
    M_rk = res['M_rk']
    M_rkf = res['M_rkf']
    M_analytical_eu = res['M_analytic_eu']
    M_analytical_rk4 = res['M_analytic_rk']
    M_analytical_rkf = res['M_analytic_rkf']

    # error array
    err_eu = M_eu - M_analytical_eu
    err_rk = M_rk - M_analytical_rk4
    err_rkf = M_rkf - M_analytical_rkf

    # global errors
    mse_global_eu = np.mean((err_eu)**2)
    mse_global_rk = np.mean((err_rk)**2)
    mse_global_rkf = np.mean((err_rkf)**2)
    
    # component errors
    mse_comp_eu = np.mean((err_eu)**2, axis=0)
    mse_comp_rk = np.mean((err_rk)**2, axis=0)
    mse_comp_rkf = np.mean((err_rkf)**2, axis=0)

    methods = [
        ('Euler', mse_global_eu, mse_comp_eu),
        ('RK4', mse_global_rk, mse_comp_rk),
        ('RKF', mse_global_rkf, mse_comp_rkf)
    ]

    components = ['Mx', 'My', 'Mz']

    for method_name, global_mse, comp_mse in methods:
        print(f"==={method_name} VS Analytic Bloch (simple case)===")
        print(f"Global MSE: {global_mse:.6e}")
        for i, comp in enumerate(components):
            print(f"{comp} MSE: {comp_mse[i]:.6e}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
  
    for component, ax in enumerate (axes):  # Mx, My, Mz
    
        ax.plot(t_eu, err_eu[:, component], 'r-', label='Euler', alpha=0.8)
        ax.plot(t_rk, err_rk[:, component], 'b-', label='RK4', alpha=0.8)
        ax.plot(t_rkf, err_rkf[:, component], 'g-', label='RKF', alpha=0.8)   

        ax.set_ylabel('Error')
        ax.set_title(f'Error over time for {components[component]}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    axes[-1].set_xlabel('Time (ms)')
    plt.tight_layout()
    plt.show()

def convergence_test():
    #convergence testing: check for which time steps makes the solution converge
    pass

if __name__ == "__main__":

    # Physical parameters
    # gamma = 2 * np.pi * 42.58e6   # rad/s/T for proton
    # B = np.array([0, 0, 0.01])       # 1 Tesla field along z
    # T1 = 1.5                      # s
    # T2 = 1e-5                     # s
    # M0 = 1.0                      # equilibrium Mz
    # M_init = np.array([1.0, 0.0, 0.0])  # start along x

    # # Simulation parameters
    # t_max = 1e-4       # 0.1 ms
    # dt = 1e-8          # 10 ns step

    compare0 = compare()
    compare1 = compare(1)
    compare2 = compare(2)
    