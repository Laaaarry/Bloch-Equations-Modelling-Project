# models/neural_bloch_gru.py
import torch
import torch.nn as nn


class NeuralBlochGRU(nn.Module):
    """
    Discrete-time GRU-based model for Bloch dynamics.

    Instead of integrating an ODE with RK4, this treats the dynamics as:
        y_{n+1} = F_theta(y_n, u_n, p, t_n)

    using a GRU cell unrolled over the time grid.

    Inputs:
        y0: (B, 3)       initial magnetization
        t:  (T,)         time grid (1D tensor)
        u:  (B, T, 4)    controls [B1x, B1y, Gx, Gy]
        p:  (B, 5)       static params [T1, T2, dB0, B0, gamma]

    Output:
        y:  (B, T, 3)    magnetization trajectory
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.hidden = hidden

        input_dim = 3 + 4 + 5 + 1  # y(3) + u(4) + p(5) + t(1) = 13

        # Project concatenated input to GRU hidden dim (with SiLU)
        self.inp = nn.Linear(input_dim, hidden)

        # GRU cell for temporal dynamics
        self.gru = nn.GRUCell(hidden, hidden)

        # Decode hidden state to magnetization
        self.out = nn.Linear(hidden, 3)

        # Nonlinearity: SiLU (a.k.a. swish) for better gradients than Tanh
        self.act = nn.SiLU()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, y0, t, u, p):
        """
        y0: (B, 3)
        t:  (T,)
        u:  (B, T, 4)
        p:  (B, 5)

        Returns:
            y: (B, T, 3)
        """
        device = y0.device
        B, T, _ = u.shape

        # Output tensor
        y = torch.empty(B, T, 3, device=device)
        y[:, 0] = y0

        # Initial hidden state: encode (y0, u0, p, t0)
        t0 = t[0]
        tb0 = torch.full((B, 1), float(t0), device=device)
        x0 = torch.cat([y0, u[:, 0, :], p, tb0], dim=-1)  # (B, 13)
        h = self.act(self.inp(x0))                        # (B, hidden_dim)

        # Optionally, we could overwrite y[:,0] with decoded h, but here we keep the true y0.

        for i in range(T - 1):
            t_n = t[i]
            tb = torch.full((B, 1), float(t_n), device=device)

            # Current input: concat current y, control, params, time
            x = torch.cat([y[:, i, :], u[:, i, :], p, tb], dim=-1)  # (B, 13)
            x = self.act(self.inp(x))                               # (B, hidden_dim)

            # GRU update
            h = self.gru(x, h)           # (B, hidden_dim)

            # Decode to next magnetization
            y[:, i + 1, :] = self.out(self.act(h))

        return y
