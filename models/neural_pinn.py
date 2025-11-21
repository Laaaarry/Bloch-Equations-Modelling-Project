# models/neural_bloch_pinn.py
import torch
import torch.nn as nn
import torch.autograd as autograd


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128, depth=4):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.Tanh())
            dim = hidden
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BlochPINN(nn.Module):
    """
    Physics-informed NN for Bloch dynamics.

    Interface matches NeuralBlochGRU:

        forward(y0, t, u, p) -> (B, T, 3)

    Inputs:
        y0 : (B, 3)      initial magnetization at t[0]
        t  : (T,)        shared time grid
        u  : (B, T, 4)   controls [B1x, B1y, Gx, Gy]
        p  : (B, 5)      [T1, T2, dB0, B0, gamma]

    Outputs:
        M : (B, T, 3)    predicted magnetization
    """

    def __init__(self, hidden: int = 128, depth: int = 4):
        super().__init__()

        # Input per (batch, time) point: t(1) + y0(3) + p(5) + u_t(4) = 13
        self.in_dim = 1 + 3 + 5 + 4
        self.out_dim = 3  # [Mx, My, Mz]

        self.mlp = MLP(self.in_dim, self.out_dim, hidden=hidden, depth=depth)

    # --------- helper: build per-time, per-batch inputs ---------
    def _build_inputs(self, y0, t, u, p, require_grad_t: bool = False):
        """
        y0 : (B, 3)
        t  : (T,)
        u  : (B, T, 4)
        p  : (B, 5)

        Returns:
            x_flat : (B*T, 13) – features for MLP
            shape_info: (B, T)
        """
        device = y0.device
        B, T, _ = u.shape

        # (T,) → (B,T,1)
        tBT = t.to(device).view(1, T, 1).expand(B, T, 1)
        # keep the flag for API compatibility, but we don't set requires_grad here;
        # we will set it directly on x_flat in physics_residual if needed.
        _ = require_grad_t  # unused, kept for signature compatibility

        y0BT = y0.view(B, 1, 3).expand(B, T, 3)
        pBT  = p.view(B, 1, 5).expand(B, T, 5)

        # x: [t, y0, p, u_t] -> (B, T, 13)
        x = torch.cat([tBT, y0BT, pBT, u], dim=-1)  # (B,T,13)

        # flatten over (B,T)
        x_flat = x.reshape(B * T, 13)               # (B*T, 13)

        return x_flat, (B, T)

    # --------- standard forward: predict M(t) --------------------
    def forward(self, y0, t, u, p):
        """
        Returns:
            M : (B, T, 3)
        """
        x_flat, (B, T) = self._build_inputs(y0, t, u, p, require_grad_t=False)
        M_flat = self.mlp(x_flat)          # (B*T, 3)
        M = M_flat.view(B, T, 3)           # (B, T, 3)
        return M

    # --------- Bloch RHS (physics) -------------------------------
    def bloch_rhs_flat(self, M_flat, u, p, shape_info):
        """
        Compute dM/dt from Bloch equations, flattened.

        M_flat : (B*T, 3)
        u      : (B, T, 4)
        p      : (B, 5)
        shape_info : (B, T)
        """
        B, T = shape_info
        device = M_flat.device

        # Unpack M
        Mx = M_flat[:, 0]
        My = M_flat[:, 1]
        Mz = M_flat[:, 2]

        # Flatten u, p
        u_flat = u.view(B * T, 4).to(device)            # (B*T, 4)
        p_flat = p.view(B, 1, 5).expand(B, T, 5) \
                    .reshape(B * T, 5).to(device)       # (B*T, 5)

        # u: [B1x, B1y, Gx, Gy]
        # p: [T1, T2, dB0, B0, gamma]
        B1x   = u_flat[:, 0]
        B1y   = u_flat[:, 1]
        dB0   = p_flat[:, 2]
        B0    = p_flat[:, 3]
        gamma = p_flat[:, 4]

        # Effective field components (ignoring gradients Gx,Gy here)
        Bx = B1x
        By = B1y
        Bz = B0 + dB0

        # Precession: gamma * (M x B)
        cx = My * Bz - Mz * By
        cy = Mz * Bx - Mx * Bz
        cz = Mx * By - My * Bx

        precess_x = gamma * cx
        precess_y = gamma * cy
        precess_z = gamma * cz

        # Relaxation terms
        T1 = p_flat[:, 0]
        T2 = p_flat[:, 1]

        # Simple choice: M0 = 1 along z
        M0 = torch.ones_like(Mz, device=device)

        relax_x = -Mx / T2
        relax_y = -My / T2
        relax_z = -(Mz - M0) / T1

        dMx = precess_x + relax_x
        dMy = precess_y + relax_y
        dMz = precess_z + relax_z

        dMdt_flat = torch.stack([dMx, dMy, dMz], dim=-1)   # (B*T, 3)
        return dMdt_flat

    # --------- physics residual for PINN loss --------------------
    def physics_residual(self, y0, t, u, p):
        """
        Compute r = dM/dt - Bloch_RHS(M, u, p) at all (B,T) points.

        Returns:
            residual : (B, T, 3)
        """
        # 1) Build inputs (time is feature 0)
        x_flat, (B, T) = self._build_inputs(y0, t, u, p, require_grad_t=False)
        # Make x_flat a leaf with grad so we can differentiate M wrt features
        x_flat = x_flat.clone().detach().requires_grad_(True)  # (B*T, 13)

        # 2) NN prediction of M(t)
        M_flat = self.mlp(x_flat)            # (B*T, 3)

        # 3) dM/dt = dM/dx[:,0] (first feature is time)
        dMdt_components = []
        for i in range(3):
            grad_full = autograd.grad(
                outputs=M_flat[:, i].sum(),
                inputs=x_flat,
                create_graph=True,
                retain_graph=True,
            )[0]                 # (B*T, 13)
            dMdt_i = grad_full[:, 0:1]  # derivative w.r.t time feature
            dMdt_components.append(dMdt_i)

        dMdt_flat = torch.cat(dMdt_components, dim=-1)  # (B*T, 3)

        # 4) Bloch RHS from physics
        rhs_flat = self.bloch_rhs_flat(M_flat, u, p, (B, T))  # (B*T, 3)

        residual_flat = dMdt_flat - rhs_flat                  # (B*T, 3)
        residual = residual_flat.view(B, T, 3)                # (B, T, 3)

        return residual
