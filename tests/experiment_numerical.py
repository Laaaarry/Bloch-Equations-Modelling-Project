# bloch equations --> dM/dt = gamma(M x B) - [Mx / T2, My, / T2, (Mz - M0) / T1]
# M(t) is a magnetization vector [Mx, My, Mz], B is magnetic field B(t)
# gamma (abreviated as ~y for now) is gyromagnetic ratio (characteristic)
# T1 and T2 are respectively longitudinal and transversal relaxation times
# M0 is equilibrium magnetization times along z-axis
# ~y(M x B) is precession term, the vector is relaxation term

# basic euler's and runge-kutta 4th order method attempt, time-invariant B
import numpy as np

def bloch_ode(t, M, gamma, T1, T2, M0, B):
    
    # M.shape = (N, 3) = (# spins, directions)

    # magnetizations in each direction
    Mx = M[:, 0]
    My = M[:, 1]
    Mz = M[:, 2]

    # precession term: ~y * (M x B)
    precession_term = gamma * np.cross(M, B) # thank you numpy

    # relaxation term: - [Mx / T2, My, / T2, (Mz - M0) / T1]
    relaxation_term = np.empty_like(M)
    relaxation_term[:, 0] = - Mx / T2
    relaxation_term[:, 1] = - My / T2
    relaxation_term[:, 2] = - (Mz - M0) / T1

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

def rkf_step(f, t, M, dt, *args):
    k1 = f(t, M, *args)
    k2 = f(t + dt/4, M + dt*k1/4, *args)
    k3 = f(t + 3*dt/8, (M + 3*dt*k1/32 + 9*dt*k2/32), *args)
    k4 = f(t + 12*dt/13, (M + 1932*dt*k1/2197 - 7200*dt*k2/2197 + 7296*dt*k3/2197), *args)
    k5 = f(t + dt, (M + 439*dt*k1/216 - 8*dt*k2 + 3680*dt*k3/513 - 845*dt*k4/4104), *args)
    k6 = f(t + dt/2, (M - 8*dt*k1/27 + 2*dt*k2 - 3544*dt*k3/2565 + 1859*dt*k4/4104 -11*dt*k5/40), *args)

    order_5 = M + dt*(16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55)
    order_4 = M + dt*(25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - 1*k5/5)
    err = np.abs(order_5-order_4) #error to calculate new time step

    return order_5, err

def simulate_bloch(method, M0_vec, t_max, dt, gamma, T1, T2, M0, B):
    t_points = np.arange(0, t_max + dt, dt)

    N = M0_vec.shape[0] # number of spins

    M = np.zeros((len(t_points), N, 3)) # solution magnetizations
    M[0] = M0_vec # set initial value 

    for i in range(len(t_points) - 1):
        t = t_points[i]
        M[i+1] = method(bloch_ode, t, M[i], dt, gamma, T1, T2, M0, B)
    return t_points, M

def bloch_rkf(method, M0_vec, t_max, dt, gamma, T1, T2, M0, B, err_tol= 1e-12):
    #Simulate Bloch equations using RKF
    t_cur = 0
    M_cur = M0_vec

    t_points = [t_cur]
    M_points = [M0_vec]

    while t_cur < t_max: 
        M_try, err = method(bloch_ode, t_cur, M_cur, dt, gamma, T1, T2, M0, B)
        err_cur = np.max(err)
        if err_cur <= err_tol:
            t_cur += dt
            M_cur = M_try
            t_points.append(t_cur)
            M_points.append(M_cur)
        else:
            dt = min(0.9*dt*(err_tol/err_cur + 1e-20)**0.2, t_max-t_cur) #calculate new step size 
    return np.array(t_points), np.array(M_points)