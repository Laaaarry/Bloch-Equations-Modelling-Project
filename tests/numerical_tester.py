import experiment_numerical as exp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import pandas as pd

def analytical_soln_simple(M_init, t, gamma, T1, T2, M0, B0, dB0):
    """
    Analytic Bloch solution for simple case:
      - no RF or gradients (u = 0)
      - constant B0 along z
      - relaxation with T1, T2
      - precession around z: omega = gamma * (B0 + dB0)
    """
    Mx0 = M_init[:, 0][None, : ] # (1, N)
    My0 = M_init[:, 1][None, : ]
    Mz0 = M_init[:, 2][None, : ]

    Mxy0 = np.sqrt(Mx0**2 + My0**2 + 1e-12)
    phi0 = np.arctan2(My0, Mx0)

    E1 = np.exp(-t[:, None] / (T1 + 1e-12)) # (N_time, 1)
    E2 = np.exp(-t[:, None] / (T2 + 1e-12))

    omega = -gamma * (B0 + dB0)
    phase = omega * t[:, None] + phi0

    # component solutions
    Mx = Mxy0 * E2 * np.cos(phase)
    My = Mxy0 * E2 * np.sin(phase)
    Mz = M0 + (Mz0 - M0) * E1

    M = np.stack([Mx, My, Mz], axis=-1) # (N_time, N, 3)
    return M

def gen_test_case(test_id = 0):
    """
    Generates test cases for analytical and numerical solution
    """

    np.random.seed(123 + test_id)
    N = 1

    # setting physical parameters
    M_init = np.random.randn(N, 3) # initial magnetization, shape (3,)
    M_init = M_init / (np.linalg.norm(M_init, axis=1, keepdims=True) + 1e-12) # normalize initial magnetization

    t_max = 10 # simulation time (seconds)
    dt = 0.01 # time step (seconds)
    gamma = 1 #2 * np.pi * 42.58e6 # proton gyromagnetic ratio (rad/s/T)

    T1 = np.random.uniform(2, 5.0, size=N) # set T1 relaxation (seconds)
    T2 = np.random.uniform(0.6, 2.0, size=N) # set T2 relaxation (seconds)

    M0 = 1.0 # equilibrium magnetization

    B0  = np.random.uniform(1.5, 3.0) # B0 static magnetic field (T)
    dB0 = np.random.uniform(-0.1, 0.1, size=N) # B0 inhomogeneity for each spin (T)

    return (M_init, t_max, dt, gamma, T1, T2, M0, B0, dB0, N)

def compare(test_id=0):
    """
    Compare numerical to analytical solution in terms of runtime and MSE
    """

    M_init, t_max, dt, gamma, T1, T2, M0, B0, dB0, N = gen_test_case(test_id)
    print(f'================TEST CASE {test_id}================')
    print(f'Parameters')
    print(f'Initial Magnetization: {M_init}')
    print(f'Simulation Time: {t_max} s')
    print(f'Time step: {dt} s')
    print(f'Gyromagnetic Ratio: {gamma} rad/s/T')
    print(f'T1 Relaxation: {T1} s')
    print(f'T2 Relaxation: {T2} s')
    print(f'Equilibrium Magnetization: {M0}')
    print(f'Static Field: {B0} T')
    print(f'B0 Inhomogeneity: {dB0} T')
    print(f'Number of Spins: {N}')
    print('____________________________________________________')

    B_field = np.zeros((N, 3))
    B_field[:, 2] = B0 + dB0

    print('Computing numerical solutions...')
    start = time.time()
    t_eu, M_eu = exp.simulate_bloch(exp.euler_step, M_init, t_max, dt, gamma, T1, T2, M0, B_field)
    end = time.time()
    runtime_eu = end-start

    start = time.time()
    t_rk, M_rk = exp.simulate_bloch(exp.rk4_step, M_init, t_max, dt, gamma, T1, T2, M0, B_field)
    end = time.time()
    runtime_rk4 = end-start

    start = time.time()
    t_rkf, M_rkf = exp.bloch_rkf(exp.rkf_step, M_init, t_max, dt, gamma, T1, T2, M0, B_field)
    end = time.time()
    runtime_rkf = end-start

    M_eu_net = np.sum(M_eu, axis=1) / N
    M_rk_net = np.sum(M_rk, axis=1) / N
    M_rkf_net = np.sum(M_rkf, axis=1) / N

    # print(M_eu.shape) #(10001, 3)
    # print(M_rk.shape) #(10001, 3)

    print(f"Euler: {len(t_eu)} points")
    print(f"RK4:   {len(t_rk)} points")
    print(f"RKF:   {len(t_rkf)} points (adaptive)")
   
    print('Computing analytical solutions...')
    
    start = time.time()
    M_analytical_eu = analytical_soln_simple(M_init, t_eu, gamma, T1, T2, M0, B0, dB0)
    end = time.time()
    runtime_eu_analytic = end-start
    
    start = time.time()
    M_analytical_rk4 = analytical_soln_simple(M_init, t_rk, gamma, T1, T2, M0, B0, dB0)
    M_analytical_rkf = analytical_soln_simple(M_init, t_rkf, gamma, T1, T2, M0, B0, dB0)
    end = time.time()
    runtime_rk4_analytic = end-start

    start = time.time()
    M_analytical_eu_net = np.sum(M_analytical_eu, axis=1) / N
    M_analytical_rk4_net = np.sum(M_analytical_rk4, axis=1) / N
    M_analytical_rkf_net = np.sum(M_analytical_rkf, axis=1) / N
    end = time.time()
    runtime_rkf_analytic = end-start

    res = {
        't_eu': t_eu, 'M_eu': M_eu_net,
        't_rk': t_rk, 'M_rk': M_rk_net,
        't_rkf': t_rkf, 'M_rkf': M_rkf_net,
        'M_analytic_eu': M_analytical_eu_net,
        'M_analytic_rk': M_analytical_rk4_net,
        'M_analytic_rkf': M_analytical_rkf_net,
        'runtime_eu': runtime_eu,
        'runtime_rk4': runtime_rk4,
        'runtime_rkf': runtime_rkf,
        'runtime_eu_analytic': runtime_eu_analytic,
        'runtime_rk4_analytic': runtime_rk4_analytic,
        'runtime_rkf_analytic': runtime_rkf_analytic
    }

    visual_solution(res)
    visual_error(res)

    return res

def visual_solution(res):
    """
    Visualize numerical and analytical solutions
    """
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
        ax3d.plot(M_analytical[:, 0], M_analytical[:, 1], M_analytical[:, 2], color='k', alpha=0.8)

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
    runtime_eu = res['runtime_eu']
    runtime_rk4 = res['runtime_rk4']
    runtime_rkf = res['runtime_rkf']
    runtime_eu_analytic = res['runtime_eu_analytic']
    runtime_rk4_analytic = res['runtime_rk4_analytic']
    runtime_rkf_analytic = res['runtime_rkf_analytic']

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

    # for method_name, global_mse, comp_mse in methods:
    #     print(f"==={method_name} VS Analytic Bloch (simple case)===")
    #     print(f"Global MSE: {global_mse:.6e}")
    #     for i, comp in enumerate(components):
    #         print(f"{comp} MSE: {comp_mse[i]:.6e}")

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

    rows = [
        {
            'Method': 'Euler',
            'Avg_MSE': mse_global_eu,
            'Mx_MSE': mse_comp_eu[0],
            'My_MSE': mse_comp_eu[1],
            'Mz_MSE': mse_comp_eu[2],
            'Runtime_s': runtime_eu,
            'Time_Points': len(t_eu)
        },
        {
            'Method': 'RK4',
            'Avg_MSE': mse_global_rk,
            'Mx_MSE': mse_comp_rk[0],
            'My_MSE': mse_comp_rk[1],
            'Mz_MSE': mse_comp_rk[2],
            'Runtime_s': runtime_rk4,
            'Time_Points': len(t_rk)
        },
        {
            'Method': 'RKF',
            'Avg_MSE': mse_global_rkf,
            'Mx_MSE': mse_comp_rkf[0],
            'My_MSE': mse_comp_rkf[1],
            'Mz_MSE': mse_comp_rkf[2],
            'Runtime_s': runtime_rkf,
            'Time_Points': len(t_rkf)
        }
    ]
    
    df = pd.DataFrame(rows)
    print(df)

if __name__ == "__main__":
    compare0 = compare()