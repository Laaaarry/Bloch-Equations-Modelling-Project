# models/neural_ode_rk4_fast.py
import torch
import torch.nn as nn

class ODEFunc(nn.Module):
    """
    f_theta(y,u,p,t): learns the vector field for Bloch dynamics.
    Input:  (B, 13) = [y(3), u(4), p(5), t(1)]
    Output: (B, 3)  = dy/dt
    """
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 3)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t, y, u_t, p):
        t = torch.as_tensor(t, dtype=y.dtype, device=y.device)
        if t.ndim > 0:
            t = t[0]
        tb = torch.full((y.shape[0], 1), float(t), device=y.device)
        x = torch.cat([y, u_t, p, tb], dim=-1)
        return self.net(x)  # (B,3)


class NeuralBlochRK4(nn.Module):
    """
    Fixed-step RK4 Neural ODE.
    Integrates y' = f_theta(y,u,p,t) over a known time grid t.
    """
    def __init__(self, hidden=128):
        super().__init__()
        self.func = ODEFunc(hidden=hidden)

    def _interp_linear_batch(self, t_grid, u_batch, tq):
        """Vectorized linear interpolation for controls."""
        idx = torch.searchsorted(t_grid, tq).clamp(1, t_grid.numel()-1)
        t0, t1 = t_grid[idx-1], t_grid[idx]
        w = (tq - t0) / (t1 - t0 + 1e-12)
        u0, u1 = u_batch[:, idx-1, :], u_batch[:, idx, :]
        return (1-w)*u0 + w*u1

    def forward(self, y0, t, u, p):
        """
        y0: (B,3), t: (T,), u: (B,T,4), p: (B,5)
        Returns y: (B,T,3)
        """
        B, T, _ = u.shape
        y = torch.empty(B, T, 3, device=y0.device)
        y[:, 0] = y0
        h = t[1] - t[0]

        for i in range(T-1):
            t_n = t[i]
            y_n = y[:, i]
            u_n = self._interp_linear_batch(t, u, t_n)

            k1 = self.func(t_n, y_n, u_n, p)
            k2 = self.func(t_n + 0.5*h, y_n + 0.5*h*k1,
                           self._interp_linear_batch(t, u, t_n + 0.5*h), p)
            k3 = self.func(t_n + 0.5*h, y_n + 0.5*h*k2,
                           self._interp_linear_batch(t, u, t_n + 0.5*h), p)
            k4 = self.func(t_n + h, y_n + h*k3,
                           self._interp_linear_batch(t, u, t_n + h), p)

            y[:, i+1] = y_n + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return y
