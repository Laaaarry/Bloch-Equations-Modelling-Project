# models/neural_ode1.py
import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    """
    f_theta(y, u, p, t): R^{3+4+5+1} -> R^3
    MLP: [13] -> 128 -> 128 -> [3], tanh activations
    Inputs per call:
      y  : (B, 3)
      u_t: (B, 4)  -- controls evaluated at scalar time t
      p  : (B, 5)  -- static params
      t  : scalar  -- may arrive as 0-D or 1-D tensor; we coerce safely
    """
    def __init__(self, hidden=128, dim_y=3, dim_u=4, dim_p=5):
        super().__init__()
        in_dim = dim_y + dim_u + dim_p + 1  # +1 for time t
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim_y),
        )
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, t, y, u_t, p):
        # Coerce t to scalar safely (handles 0-D or 1-D tensors)
        if isinstance(t, torch.Tensor):
            t_scalar = t.item() if t.ndim == 0 else t.reshape(-1)[0].item()
        else:
            t_scalar = float(t)
        tb = torch.full((y.shape[0], 1), t_scalar, device=y.device, dtype=y.dtype)
        x = torch.cat([y, u_t, p, tb], dim=-1)  # (B, 13)
        return self.net(x)  # (B, 3)


class NeuralBloch(nn.Module):
    """
    Neural ODE wrapper integrating:
        y' = f_theta(y, u(t), p, t)
    over a shared time grid t.

    Expected shapes:
      y0: (B, 3)
      t : (T,)  (monotonic increasing)
      u : (B, T, 4)
      p : (B, 5)

    Returns:
      y:  (B, T, 3)
    """
    def __init__(self, hidden=128):
        super().__init__()
        self.func = ODEFunc(hidden=hidden)

    @staticmethod
    def _interp_linear_1d(t_grid, u_seq, tq):
        """
        Linear interpolation of u at scalar time tq.
        t_grid: (T,), u_seq: (T, D), tq: scalar tensor
        returns: (D,)
        """
        # search index of right bin, clamp to [1, T-1]
        idx = torch.searchsorted(t_grid, tq).clamp(1, t_grid.numel() - 1)
        t0, t1 = t_grid[idx - 1], t_grid[idx]
        w = (tq - t0) / (t1 - t0 + 1e-12)
        return (1.0 - w) * u_seq[idx - 1] + w * u_seq[idx]

    def forward(self, y0, t, u, p, rtol=1e-5, atol=1e-6):
        """
        Integrate from y0 along t with controls u and params p.
        """
        B, T, _ = u.shape
        y0_flat = y0.reshape(-1)  # (B*3,)

        # Right-hand side for odeint (expects flat state)
        def rhs(tq, y_flat):
            y = y_flat.view(B, 3)  # (B,3)
            # Batch interpolate controls at tq
            u_t = torch.stack([self._interp_linear_1d(t, u[b], tq) for b in range(B)], dim=0)  # (B,4)
            dy = self.func(tq, y, u_t, p)  # (B,3)
            return dy.view(-1)

        # Integrate over provided time grid
        yT = odeint(rhs, y0_flat, t, method="dopri5", rtol=rtol, atol=atol)  # (T, B*3)
        yT = yT.view(len(t), B, 3).transpose(0, 1).contiguous()              # (B, T, 3)
        return yT


