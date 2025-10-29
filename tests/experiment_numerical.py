# bloch equations --> dM/dt = gamma(M x B) - [Mx / T2, My, / T2, (Mz - M0) / T1]
# M(t) is a magnetization vector [Mx, My, Mz], B is magnetic field B(t)
# gamma (abreviated as ~y for now) is gyromagnetic ratio (characteristic)
# T1 and T2 are respectively longitudinal and transversal relaxation times
# M0 is equilibrium magnetization times along z-axis
# ~y(M x B) is precession term, the vector is relaxation term

# basic euler's and runge-kutta 4th order method attempt, time-invariant B

import numpy as np

def bloch_ode(t, M, gamma, T1, T2, M0, B):
    # type - floats
    # t = time
    # gamma = gyromagnetic ratio (~y)
    # T1, T2 = relaxation times
    # M0 = equilibrium magnetization
    # type - arrays
    # M = M(t) = magnetization vector - 3 components
    # B = B(t) = magnetic field - 3 components

    # t isn't used since for this case, we assume B doesn't change with time

    # intermediate variables:
    Mx, My, Mz = M
    Bx, By, Bz = B

    # precession term: ~y * (M x B)
    precession_term = gamma * np.cross(M, B) # thank you numpy

    # relaxation term: - [Mx / T2, My, / T2, (Mz - M0) / T1]
    relaxation_term = np.array([
        -Mx / T2,
        -My / T2,
        -(Mz - M0) / T1
    ])

    # return result
    return precession_term + relaxation_term

def euler_step(f, t, M, dt, *args):
    # 1 euler step
    # dt is time step
    return M + dt * f(t, M, *args)


def rk4_step(f, t, M, dt, *args):
    # 1 rk4 step
    # dt is time step
    k1 = f(t, M, *args)
    k2 = f(t + dt/2, M + dt*k1/2, *args)
    k3 = f(t + dt/2, M + dt*k2/2, *args)
    k4 = f(t + dt, M + dt*k3, *args)
    return M + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


def simulate_bloch(method, M0_vec, t_max, dt, gamma, T1, T2, M0, B):
    #Simulate Bloch equations using Euler or RK4.
    # method: function = "euler_step" or "rk4_step"
    # M0: array/vector --> [Mx0, My0, Mz0]
    # t_max: float --> simulation time (s)
    # dt: float --> time step (s)

    t_points = np.arange(0, t_max + dt, dt)
    M = np.zeros((len(t_points), 3))
    M[0] = M0_vec

    for i in range(len(t_points) - 1):
        t = t_points[i]
        M[i+1] = method(bloch_ode, t, M[i], dt, gamma, T1, T2, M0, B)
    return t_points, M

