import torch
import torch.nn as nn
import numpy as np
import math


# ============================================================================
# Utility: Analytic Bloch RHS
# ============================================================================

def bloch_rhs_analytic(M, u, p, M_eq: float = 1.0):
    while p.ndim < M.ndim:
        p = p.unsqueeze(1)

    T1 = p[..., 0]
    T2 = p[..., 1]
    dB0 = p[..., 2]
    B0 = p[..., 3]
    gamma = p[..., 4]

    B1x = u[..., 0]
    B1y = u[..., 1]

    # Make Bz the same shape as B1x/B1y (so torch.stack works)
    Bz = (B0 + dB0)
    if Bz.shape != B1x.shape:
        Bz = Bz.expand_as(B1x)  # or: Bz = Bz * torch.ones_like(B1x)

    B_eff = torch.stack([B1x, B1y, Bz], dim=-1)

    precession = gamma.unsqueeze(-1) * torch.cross(M, B_eff, dim=-1)

    Mx, My, Mz = M[..., 0], M[..., 1], M[..., 2]
    dMx_relax = -Mx / T2
    dMy_relax = -My / T2
    dMz_relax = (M_eq - Mz) / T1
    relaxation = torch.stack([dMx_relax, dMy_relax, dMz_relax], dim=-1)

    return precession + relaxation



# ============================================================================
# Helper Modules
# ============================================================================

class FourierTimeEncoder(nn.Module):
    """Learnable Fourier features for time encoding."""
    
    def __init__(self, num_features: int = 16):
        super().__init__()
        self.num_features = num_features
        # Learnable frequencies
        self.frequencies = nn.Parameter(torch.randn(num_features) * 0.1)
    
    def forward(self, t):
        """
        t: (B,) or scalar
        Returns: (B, 2*num_features)
        """
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=self.frequencies.dtype, device=self.frequencies.device)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        
        t = t.unsqueeze(-1)  # (B, 1)
        freqs = self.frequencies.unsqueeze(0)  # (1, num_features)
        
        angles = 2 * math.pi * t * freqs  # (B, num_features)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class ResidualBlock(nn.Module):
    """Residual block with layer norm."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, x):
        return x + self.layers(x)


# ============================================================================
# Model 1: Enhanced UDE with Better Architecture
# ============================================================================

class EnhancedBlochUDE(nn.Module):
    """
    Improved UDE with:
    - Larger capacity (deeper, wider)
    - Input normalization
    - Fourier time encoding
    - Separate control and parameter processing
    - Skip connections
    
    This is the best starting point for most use cases.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 5,
        residual_scale: float = 0.3,
        n_substeps: int = 2,
        fourier_features: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.n_substeps = n_substeps
        self.fourier_features = fourier_features
        
        # Learnable Fourier features for time encoding
        self.time_encoder = FourierTimeEncoder(fourier_features)
        
        # Control encoder (process RF + gradients)
        self.control_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        
        # Parameter encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
        )
        
        # Main network
        in_dim = 3 + 2*fourier_features + 64 + 32  # M + time + controls + params
        
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        for i in range(num_layers - 1):
            layers.append(ResidualBlock(hidden_dim, dropout))
        
        layers.append(nn.Linear(hidden_dim, 3))
        layers.append(nn.Tanh())  # Bounded corrections
        
        self.nn = nn.Sequential(*layers)
        
        # Input normalization
        self.M_norm = nn.LayerNorm(3)
    
    def _rhs_total(self, M, u, p, t_scalar):
        """Total RHS with analytic + learned correction."""
        device = M.device
        B = M.shape[0]
        
        # Analytic term
        rhs_analytic = bloch_rhs_analytic(M, u, p, M_eq=1.0)
        
        # Normalize inputs
        M_normed = self.M_norm(M)
        
        # Encode time
        if not torch.is_tensor(t_scalar):
            t_scalar = torch.tensor(t_scalar, dtype=M.dtype, device=device)
        if t_scalar.ndim == 0:
            t_scalar = t_scalar.expand(B)
        t_enc = self.time_encoder(t_scalar)  # (B, 2*fourier_features)
        
        # Encode controls and parameters
        u_enc = self.control_encoder(u)  # (B, 64)
        p_enc = self.param_encoder(p)   # (B, 32)
        
        # Concatenate all features
        nn_in = torch.cat([M_normed, t_enc, u_enc, p_enc], dim=-1)
        
        # Learned correction
        rhs_nn = self.nn(nn_in)
        
        return rhs_analytic + self.residual_scale * rhs_nn
    
    def dynamics(self, t, M, u, p):
        """
        For Bloch RHS matching loss.
        t: (N,) or scalar
        M: (N, 3)
        u: (N, 4)
        p: (N, 5)
        Returns: (N, 3)
        """
        return self._rhs_total(M, u, p, t)
    
    def forward(self, M0, t_grid, u, p):
        """
        M0: (B, 3) initial state
        t_grid: (L,) time points
        u: (B, L, 4) controls
        p: (B, 5) parameters
        Returns: (B, L, 3) magnetization trajectory
        """
        device = M0.device
        B, L, _ = u.shape
        
        M = torch.zeros(B, L, 3, device=device, dtype=M0.dtype)
        M[:, 0, :] = M0
        
        base_dt = (t_grid[1] - t_grid[0]).to(device)
        dt = base_dt / float(self.n_substeps)
        
        for k in range(L - 1):
            Mk = M[:, k, :].clone()   # <-- critical change
            uk = u[:, k, :]
            pk = p
            t0 = t_grid[k].to(device)

            for _ in range(self.n_substeps):
                k1 = self._rhs_total(Mk, uk, pk, t0)
                k2 = self._rhs_total(Mk + 0.5*dt*k1, uk, pk, t0 + 0.5*dt)
                k3 = self._rhs_total(Mk + 0.5*dt*k2, uk, pk, t0 + 0.5*dt)
                k4 = self._rhs_total(Mk + dt*k3, uk, pk, t0 + dt)

                Mk = Mk + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                t0 = t0 + dt

            M[:, k + 1, :] = Mk

        return M


# ============================================================================
# Model 2: Latent Neural ODE (Encoder-ODE-Decoder)
# ============================================================================

class LatentODEFunc(nn.Module):
    """ODE function in latent space."""
    
    def __init__(
        self,
        latent_dim: int,
        control_dim: int,
        param_dim: int,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()
        
        in_dim = latent_dim + control_dim + param_dim
        
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, z, u, p, t):
        """
        z: (B, latent_dim)
        u: (B, control_dim)
        p: (B, param_dim)
        t: scalar or (B,)
        Returns: (B, latent_dim)
        """
        ode_in = torch.cat([z, u, p], dim=-1)
        return self.net(ode_in)


class LatentBlochODE(nn.Module):
    """
    Encode initial state + controls into latent space,
    evolve with Neural ODE, then decode.
    
    Best for long sequences - compresses dynamics to reduce error accumulation.
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        num_ode_layers: int = 4,
        n_substeps: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_substeps = n_substeps
        
        # Encoder: (M0, controls summary, params) -> latent z0
        self.encoder = nn.Sequential(
            nn.Linear(3 + 64 + 5, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, latent_dim),
        )
        
        # Control summarizer (process full sequence)
        self.control_summarizer = nn.GRU(4, 32, batch_first=True, num_layers=2)
        
        # ODE function in latent space
        self.ode_net = LatentODEFunc(
            latent_dim=latent_dim,
            control_dim=4,
            param_dim=5,
            hidden_dim=hidden_dim,
            num_layers=num_ode_layers,
        )
        
        # Decoder: latent -> M
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 3),
        )
    
    def dynamics(self, t, z, u, p):
        """For compatibility with Bloch loss (operates in latent space)."""
        return self.ode_net(z, u, p, t)
    
    def forward(self, M0, t_grid, u, p):
        """
        M0: (B, 3)
        t_grid: (L,)
        u: (B, L, 4)
        p: (B, 5)
        Returns: (B, L, 3)
        """
        device = M0.device
        B, L, _ = u.shape
        
        # Summarize control sequence
        _, u_hidden = self.control_summarizer(u)  # (2, B, 32)
        u_summary = u_hidden[-1]  # (B, 32)
        u_summary = torch.cat([u_summary, torch.zeros(B, 32, device=device)], dim=-1)
        
        # Encode initial condition
        z0 = self.encoder(torch.cat([M0, u_summary, p], dim=-1))
        
        # Evolve latent state
        z = torch.zeros(B, L, self.latent_dim, device=device, dtype=M0.dtype)
        z[:, 0, :] = z0
        
        base_dt = (t_grid[1] - t_grid[0]).to(device)
        dt = base_dt / float(self.n_substeps)
        
        for k in range(L - 1):
            zk = z[:, k, :]
            uk = u[:, k, :]
            pk = p
            t0 = t_grid[k].to(device)
            
            for _ in range(self.n_substeps):
                # RK4 in latent space
                k1 = self.ode_net(zk, uk, pk, t0)
                k2 = self.ode_net(zk + 0.5*dt*k1, uk, pk, t0 + 0.5*dt)
                k3 = self.ode_net(zk + 0.5*dt*k2, uk, pk, t0 + 0.5*dt)
                k4 = self.ode_net(zk + dt*k3, uk, pk, t0 + dt)
                
                zk = zk + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                t0 = t0 + dt
            
            z[:, k + 1, :] = zk
        
        # Decode all time steps
        M = self.decoder(z.reshape(-1, self.latent_dim)).reshape(B, L, 3)
        
        return M


# ============================================================================
# Model 3: Augmented Neural ODE
# ============================================================================

class AugmentedBlochODE(nn.Module):
    """
    Augmented Neural ODE: adds extra dimensions to state space
    to increase expressivity without changing physics dimensions.
    
    State: [Mx, My, Mz, a1, a2, ..., a_k]
    Only first 3 dims are returned as magnetization.
    
    Best for complex RF pulse shapes that need more expressivity.
    """
    
    def __init__(
        self,
        aug_dim: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 4,
        n_substeps: int = 2,
        fourier_features: int = 12,
    ):
        super().__init__()
        self.aug_dim = aug_dim
        self.n_substeps = n_substeps
        
        state_dim = 3 + aug_dim  # Augmented state
        
        self.time_encoder = FourierTimeEncoder(fourier_features)
        
        # ODE function for augmented state
        in_dim = state_dim + 2*fourier_features + 4 + 5  # state + time + u + p
        
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers):
            layers.append(ResidualBlock(hidden_dim, dropout=0.1))
        
        layers.append(nn.Linear(hidden_dim, state_dim))
        
        self.ode_net = nn.Sequential(*layers)
    
    def _rhs(self, state, u, p, t_scalar):
        """RHS for augmented state."""
        device = state.device
        B = state.shape[0]
        
        # Time encoding
        if not torch.is_tensor(t_scalar):
            t_scalar = torch.tensor(t_scalar, dtype=state.dtype, device=device)
        if t_scalar.ndim == 0:
            t_scalar = t_scalar.expand(B)
        t_enc = self.time_encoder(t_scalar)
        
        # Concatenate everything
        ode_in = torch.cat([state, t_enc, u, p], dim=-1)
        
        return self.ode_net(ode_in)
    
    def dynamics(self, t, state, u, p):
        """For Bloch loss (operates on augmented state)."""
        return self._rhs(state, u, p, t)
    
    def forward(self, M0, t_grid, u, p):
        """
        M0: (B, 3)
        t_grid: (L,)
        u: (B, L, 4)
        p: (B, 5)
        Returns: (B, L, 3)
        """
        device = M0.device
        B, L, _ = u.shape
        
        # Initialize augmented state (zeros for augmented dims)
        state = torch.zeros(B, L, 3 + self.aug_dim, device=device, dtype=M0.dtype)
        state[:, 0, :3] = M0
        
        base_dt = (t_grid[1] - t_grid[0]).to(device)
        dt = base_dt / float(self.n_substeps)
        
        for k in range(L - 1):
            sk = state[:, k, :]
            uk = u[:, k, :]
            pk = p
            t0 = t_grid[k].to(device)
            
            for _ in range(self.n_substeps):
                # RK4
                k1 = self._rhs(sk, uk, pk, t0)
                k2 = self._rhs(sk + 0.5*dt*k1, uk, pk, t0 + 0.5*dt)
                k3 = self._rhs(sk + 0.5*dt*k2, uk, pk, t0 + 0.5*dt)
                k4 = self._rhs(sk + dt*k3, uk, pk, t0 + dt)
                
                sk = sk + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                t0 = t0 + dt
            
            state[:, k + 1, :] = sk
        
        # Extract only magnetization (first 3 dims)
        M = state[:, :, :3]
        
        return M


# ============================================================================
# Model 4: Hybrid Physics-ML with Attention over Controls
# ============================================================================

class AttentionBlochODE(nn.Module):
    """
    Uses attention mechanism to process control sequence,
    allowing model to focus on relevant RF pulse features.
    
    Best for understanding which RF characteristics drive dynamics.
    More interpretable than other models.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        residual_scale: float = 0.3,
        n_substeps: int = 2,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.n_substeps = n_substeps
        self.hidden_dim = hidden_dim
        
        # Control attention encoder
        self.control_proj = nn.Linear(4, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*2,
            batch_first=True,
            dropout=0.1,
        )
        self.control_attention = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Per-timestep correction network
        self.correction_net = nn.Sequential(
            nn.Linear(3 + hidden_dim + 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),
        )
    
    def dynamics(self, t, M, u, p):
        """
        For Bloch loss. Note: doesn't use attention context here.
        t: (N,) or scalar
        M: (N, 3)
        u: (N, 4)
        p: (N, 5)
        """
        # Fallback to analytic + simple correction without full attention
        rhs_analytic = bloch_rhs_analytic(M, u, p, M_eq=1.0)
        
        # Simple correction without attention context
        simple_in = torch.cat([M, u, p], dim=-1)
        # Use a subset of correction_net if needed, or just return analytic
        return rhs_analytic  # Simplified for dynamics call
    
    def forward(self, M0, t_grid, u, p):
        """
        M0: (B, 3)
        t_grid: (L,)
        u: (B, L, 4)
        p: (B, 5)
        Returns: (B, L, 3)
        """
        device = M0.device
        B, L, _ = u.shape
        
        # Process control sequence with attention
        u_proj = self.control_proj(u)  # (B, L, hidden_dim)
        u_context = self.control_attention(u_proj)  # (B, L, hidden_dim)
        
        M = torch.zeros(B, L, 3, device=device, dtype=M0.dtype)
        M[:, 0, :] = M0
        
        base_dt = (t_grid[1] - t_grid[0]).to(device)
        dt = base_dt / float(self.n_substeps)
        
        for k in range(L - 1):
            Mk = M[:, k, :]
            uk = u[:, k, :]
            uk_context = u_context[:, k, :]
            pk = p
            
            for _ in range(self.n_substeps):
                # Analytic term
                rhs_analytic = bloch_rhs_analytic(Mk, uk, pk, M_eq=1.0)
                
                # Learned correction using control context
                correction_in = torch.cat([Mk, uk_context, pk], dim=-1)
                rhs_correction = self.correction_net(correction_in)
                
                # Combined RHS
                rhs = rhs_analytic + self.residual_scale * rhs_correction
                
                # Euler step (can upgrade to RK4)
                Mk = Mk + dt * rhs
            
            M[:, k + 1, :] = Mk
        
        return M


B, L = 4, 100
M0 = torch.randn(B, 3)
t_grid = torch.linspace(0, 1, L)
u = torch.randn(B, L, 4)
p = torch.abs(torch.randn(B, 5))  # Ensure positive T1, T2, etc.
    
print("Testing all four models...\n")
    
    # Model 1: Enhanced UDE
print("="*60)
print("Model 1: EnhancedBlochUDE")
print("="*60)
model1 = EnhancedBlochUDE(hidden_dim=256, num_layers=5)
n_params1 = sum(p.numel() for p in model1.parameters() if p.requires_grad)
print(f"Parameters: {n_params1:,}")
y1 = model1(M0, t_grid, u, p)
print(f"Output shape: {y1.shape}")
print(f"Output range: [{y1.min():.3f}, {y1.max():.3f}]")
print("✓ Success!\n")

# Model 2: Latent ODE
print("="*60)
print("Model 2: LatentBlochODE")
print("="*60)
model2 = LatentBlochODE(latent_dim=32, hidden_dim=256)
n_params2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
print(f"Parameters: {n_params2:,}")
y2 = model2(M0, t_grid, u, p)
print(f"Output shape: {y2.shape}")
print(f"Output range: [{y2.min():.3f}, {y2.max():.3f}]")
print("✓ Success!\n")

# Model 3: Augmented ODE
print("="*60)
print("Model 3: AugmentedBlochODE")
print("="*60)
model3 = AugmentedBlochODE(aug_dim=5, hidden_dim=256)
n_params3 = sum(p.numel() for p in model3.parameters() if p.requires_grad)
print(f"Parameters: {n_params3:,}")
y3 = model3(M0, t_grid, u, p)
print(f"Output shape: {y3.shape}")
print(f"Output range: [{y3.min():.3f}, {y3.max():.3f}]")
print("✓ Success!\n")

# Model 4: Attention ODE
print("="*60)
print("Model 4: AttentionBlochODE")
print("="*60)
model4 = AttentionBlochODE(hidden_dim=256, num_heads=4)
n_params4 = sum(p.numel() for p in model4.parameters() if p.requires_grad)
print(f"Parameters: {n_params4:,}")
y4 = model4(M0, t_grid, u, p)
print(f"Output shape: {y4.shape}")
print(f"Output range: [{y4.min():.3f}, {y4.max():.3f}]")
print("✓ Success!\n")

print("="*60)
print("All models executed successfully!")
print("="*60)


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import csv
from tqdm import tqdm
import numpy as np
from typing import Optional, Tuple, Dict, Any


# ============================================================================
# Enhanced Loss Functions
# ============================================================================

def bloch_rhs_analytic(M, u, p, M_eq: float = 1.0):
    while p.ndim < M.ndim:
        p = p.unsqueeze(1)

    T1 = p[..., 0]
    T2 = p[..., 1]
    dB0 = p[..., 2]
    B0 = p[..., 3]
    gamma = p[..., 4]

    B1x = u[..., 0]
    B1y = u[..., 1]

    # Make Bz the same shape as B1x/B1y (so torch.stack works)
    Bz = (B0 + dB0)
    if Bz.shape != B1x.shape:
        Bz = Bz.expand_as(B1x)  # or: Bz = Bz * torch.ones_like(B1x)

    B_eff = torch.stack([B1x, B1y, Bz], dim=-1)

    precession = gamma.unsqueeze(-1) * torch.cross(M, B_eff, dim=-1)

    Mx, My, Mz = M[..., 0], M[..., 1], M[..., 2]
    dMx_relax = -Mx / T2
    dMy_relax = -My / T2
    dMz_relax = (M_eq - Mz) / T1
    relaxation = torch.stack([dMx_relax, dMy_relax, dMz_relax], dim=-1)

    return precession + relaxation



class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss that emphasizes different aspects of the trajectory.
    """
    
    def __init__(
        self,
        w_full: float = 1.0,
        w_checkpoints: float = 0.5,
        w_signal: float = 0.3,
        w_endpoint: float = 0.2,
        component_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.w_full = w_full
        self.w_checkpoints = w_checkpoints
        self.w_signal = w_signal
        self.w_endpoint = w_endpoint
        
        # Default: emphasize transverse components (Mx, My) over Mz
        if component_weights is None:
            component_weights = torch.tensor([5.0, 5.0, 1.0])
        self.register_buffer('component_weights', component_weights.view(1, 1, 3))
    
    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            yhat: (B, L, 3) predicted
            y: (B, L, 3) ground truth
        
        Returns:
            loss: scalar
            loss_dict: breakdown of loss components
        """
        B, L, _ = yhat.shape
        
        # 1. Full trajectory MSE with component weighting
        diff2 = (yhat - y).pow(2)
        loss_full = (diff2 * self.component_weights).mean()
        
        # 2. Key checkpoint errors (beginning, 1/4, 1/2, 3/4, end)
        indices = [0, max(1, L//4), L//2, max(L//2+1, 3*L//4), L-1]
        checkpoint_diff = (yhat[:, indices] - y[:, indices]).pow(2)
        loss_checkpoints = (checkpoint_diff * self.component_weights).mean()
        
        # 3. Transverse magnetization (MRI signal)
        Mxy_pred = torch.sqrt(yhat[:, :, 0]**2 + yhat[:, :, 1]**2 + 1e-8)
        Mxy_true = torch.sqrt(y[:, :, 0]**2 + y[:, :, 1]**2 + 1e-8)
        loss_signal = (Mxy_pred - Mxy_true).pow(2).mean()
        
        # 4. Endpoint accuracy (final state matters for sequences)
        loss_endpoint = (yhat[:, -1] - y[:, -1]).pow(2).mean()
        
        # Total loss
        loss = (
            self.w_full * loss_full +
            self.w_checkpoints * loss_checkpoints +
            self.w_signal * loss_signal +
            self.w_endpoint * loss_endpoint
        )
        
        loss_dict = {
            'full': loss_full.item(),
            'checkpoints': loss_checkpoints.item(),
            'signal': loss_signal.item(),
            'endpoint': loss_endpoint.item(),
        }
        
        return loss, loss_dict


class PhysicsLoss(nn.Module):
    """
    Physics-based regularization losses.
    """
    
    def __init__(
        self,
        lambda_fd: float = 0.01,
        lambda_norm: float = 1e-3,
        lambda_monotonic: float = 1e-4,
        stride_fd: int = 1,
    ):
        super().__init__()
        self.lambda_fd = lambda_fd
        self.lambda_norm = lambda_norm
        self.lambda_monotonic = lambda_monotonic
        self.stride_fd = stride_fd
    
    def forward(
        self,
        yhat: torch.Tensor,
        y: torch.Tensor,
        t_grid: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            yhat: (B, L, 3) predicted
            y: (B, L, 3) ground truth
            t_grid: (L,) time grid
        
        Returns:
            loss: scalar
            loss_dict: breakdown
        """
        device = yhat.device
        t_grid = t_grid.to(device)
        B, L, _ = yhat.shape
        
        losses = {}
        
        # 1. Finite difference derivative matching
        if self.lambda_fd > 0:
            idx0 = torch.arange(0, L-1, self.stride_fd, device=device)
            idx1 = idx0 + 1
            
            dt = (t_grid[idx1] - t_grid[idx0]).view(1, -1, 1)
            
            dyhat_dt = (yhat[:, idx1] - yhat[:, idx0]) / dt
            dy_dt = (y[:, idx1] - y[:, idx0]) / dt
            
            losses['fd'] = (dyhat_dt - dy_dt).pow(2).mean()
        else:
            losses['fd'] = torch.tensor(0.0, device=device)
        
        # 2. Magnetization norm should stay close to 1 (physical constraint)
        if self.lambda_norm > 0:
            mag = yhat.norm(dim=-1)
            losses['norm'] = (mag - 1.0).pow(2).mean()
        else:
            losses['norm'] = torch.tensor(0.0, device=device)
        
        # 3. Relaxation should be monotonic for certain components
        # (Mz should recover toward equilibrium, |Mxy| should decay without RF)
        if self.lambda_monotonic > 0:
            # Check if transverse magnetization is decaying when it should
            Mxy_mag = torch.sqrt(yhat[:, :, 0]**2 + yhat[:, :, 1]**2 + 1e-8)
            
            # Penalize increases in |Mxy| between distant timesteps (crude check)
            step = max(1, L // 20)
            for i in range(0, L - step, step):
                diff = Mxy_mag[:, i+step] - Mxy_mag[:, i]
                # Only penalize if it's increasing when it shouldn't
                losses['monotonic'] = torch.relu(diff).pow(2).mean()
        else:
            losses['monotonic'] = torch.tensor(0.0, device=device)
        
        # Total
        total = (
            self.lambda_fd * losses['fd'] +
            self.lambda_norm * losses['norm'] +
            self.lambda_monotonic * losses['monotonic']
        )
        
        return total, {k: v.item() for k, v in losses.items()}


# ============================================================================
# Enhanced Training Loop
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    t_shared: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    data_loss_fn: MultiScaleLoss,
    physics_loss_fn: PhysicsLoss,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    scaler: Optional[GradScaler] = None,
    subwindow_length: Optional[int] = None,
    lambda_physics: float = 0.1,
    lambda_bloch: float = 0.01,
    accumulation_steps: int = 1,
) -> Tuple[float, torch.Tensor, Dict[str, float]]:
    """
    Enhanced training loop with:
    - Multi-scale losses
    - Physics regularization
    - Optional Bloch RHS matching
    - Gradient accumulation
    - Better logging
    """
    model.train()
    
    total_loss = 0.0
    total_elems = 0
    total_comp = torch.zeros(3, device=device)
    
    # Accumulators for detailed logging
    accum_data = {'full': 0, 'checkpoints': 0, 'signal': 0, 'endpoint': 0}
    accum_phys = {'fd': 0, 'norm': 0, 'monotonic': 0}
    accum_bloch = 0
    n_batches = 0
    
    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch+1:3d}/{total_epochs} [Train]",
        leave=True,
        ncols=140,
        position=0,
        colour='blue'
    )
    
    for batch_idx, batch in enumerate(pbar):
        y, y0, u, p = [
            batch[k].to(device, non_blocking=True)
            for k in ("y", "y0", "u", "p")
        ]
        B, T, _ = y.shape
        
        # ----- Random subwindow sampling (curriculum: start small, grow) -----
        if subwindow_length is not None and subwindow_length < T:
            L = subwindow_length
            max_start = T - L
            s = torch.randint(0, max_start + 1, (1,), device=device).item()
            e = s + L
            
            t_win = t_shared[s:e]
            u_win = u[:, s:e, :]
            y_win = y[:, s:e, :]
            M0_win = y[:, s, :].clone()

        else:
            t_win = t_shared
            u_win = u
            y_win = y
            M0_win = y0
            L = T
        
        # ----- Forward pass -----
        use_amp = scaler is not None
        
        if use_amp:
            with autocast():
                yhat = model(M0_win, t_win, u_win, p)
                
                # Data loss
                data_loss, data_dict = data_loss_fn(yhat, y_win)
                
                # Physics loss
                phys_loss, phys_dict = physics_loss_fn(yhat, y_win, t_win)
                
                # Bloch RHS matching (optional)
                if lambda_bloch > 0 and hasattr(model, 'dynamics'):
                    stride_b = max(1, L // 50)  # Sample ~50 points
                    idx = torch.arange(0, L, stride_b, device=device)
                    
                    M_sub = y_win[:, idx, :]
                    u_sub = u_win[:, idx, :]
                    t_sub = t_win[idx]
                    
                    # True Bloch RHS
                    dMdt_true = bloch_rhs_analytic(M_sub, u_sub, p)
                    
                    # Predicted RHS
                    B_, K_, _ = M_sub.shape
                    M_flat = M_sub.reshape(B_ * K_, 3)
                    u_flat = u_sub.reshape(B_ * K_, 4)
                    p_flat = p.unsqueeze(1).expand(B_, K_, -1).reshape(B_ * K_, 5)
                    t_flat = t_sub.unsqueeze(0).expand(B_, K_).reshape(-1)
                    
                    dMdt_pred = model.dynamics(t_flat, M_flat, u_flat, p_flat)
                    dMdt_pred = dMdt_pred.view(B_, K_, 3)
                    
                    bloch_loss = (dMdt_pred - dMdt_true).pow(2).mean()
                else:
                    bloch_loss = torch.tensor(0.0, device=device)
                
                # Total loss
                loss = data_loss + lambda_physics * phys_loss + lambda_bloch * bloch_loss
                loss = loss / accumulation_steps
        else:
            yhat = model(M0_win, t_win, u_win, p)
            
            data_loss, data_dict = data_loss_fn(yhat, y_win)
            phys_loss, phys_dict = physics_loss_fn(yhat, y_win, t_win)
            
            if lambda_bloch > 0 and hasattr(model, 'dynamics'):
                stride_b = max(1, L // 50)
                idx = torch.arange(0, L, stride_b, device=device)
                
                M_sub = y_win[:, idx, :]
                u_sub = u_win[:, idx, :]
                t_sub = t_win[idx]
                
                dMdt_true = bloch_rhs_analytic(M_sub, u_sub, p)
                
                B_, K_, _ = M_sub.shape
                M_flat = M_sub.reshape(B_ * K_, 3)
                u_flat = u_sub.reshape(B_ * K_, 4)
                p_flat = p.unsqueeze(1).expand(B_, K_, -1).reshape(B_ * K_, 5)
                t_flat = t_sub.unsqueeze(0).expand(B_, K_).reshape(-1)
                
                dMdt_pred = model.dynamics(t_flat, M_flat, u_flat, p_flat)
                dMdt_pred = dMdt_pred.view(B_, K_, 3)
                
                bloch_loss = (dMdt_pred - dMdt_true).pow(2).mean()
            else:
                bloch_loss = torch.tensor(0.0, device=device)
            
            loss = data_loss + lambda_physics * phys_loss + lambda_bloch * bloch_loss
            loss = loss / accumulation_steps
        
        # ----- Backward pass -----
        if use_amp:
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        # ----- Logging -----
        batch_elems = y.numel()
        total_loss += loss.item() * accumulation_steps * batch_elems
        total_elems += batch_elems
        
        diff2 = (yhat - y_win).pow(2)
        total_comp += diff2.sum(dim=(0, 1))
        
        # Accumulate detailed losses
        for k, v in data_dict.items():
            accum_data[k] += v
        for k, v in phys_dict.items():
            accum_phys[k] += v
        accum_bloch += bloch_loss.item()
        n_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4e}',
            'data': f'{data_loss.item():.4e}',
            'phys': f'{phys_loss.item():.4e}',
            'bloch': f'{bloch_loss.item():.4e}',
        })
    
    # Final averages
    mean_loss = total_loss / max(total_elems, 1)
    num_positions = total_elems / 3.0 if total_elems > 0 else 1.0
    mean_comp = total_comp / num_positions
    
    # Detailed loss breakdown
    loss_breakdown = {
        'data': {k: v / n_batches for k, v in accum_data.items()},
        'physics': {k: v / n_batches for k, v in accum_phys.items()},
        'bloch': accum_bloch / n_batches,
    }
    
    return mean_loss, mean_comp.cpu(), loss_breakdown


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    t_shared: torch.Tensor,
    data_loss_fn: MultiScaleLoss,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, torch.Tensor, Dict[str, float]]:
    """Enhanced evaluation with detailed metrics."""
    model.eval()
    
    total_loss = 0.0
    total_elems = 0
    total_comp = torch.zeros(3, device=device)
    
    accum_data = {'full': 0, 'checkpoints': 0, 'signal': 0, 'endpoint': 0}
    n_batches = 0
    
    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch+1:3d}/{total_epochs} [Val]  ",
        leave=True,
        ncols=140,
        position=0,
        colour='green'
    )
    
    for batch in pbar:
        y, y0, u, p = [
            batch[k].to(device, non_blocking=True)
            for k in ("y", "y0", "u", "p")
        ]
        
        use_amp = torch.cuda.is_available() and device.type == "cuda"
        
        if use_amp:
            with autocast():
                yhat = model(y0, t_shared, u, p)
                loss, data_dict = data_loss_fn(yhat, y)
        else:
            yhat = model(y0, t_shared, u, p)
            loss, data_dict = data_loss_fn(yhat, y)
        
        batch_elems = y.numel()
        total_loss += loss.item() * batch_elems
        total_elems += batch_elems
        
        diff2 = (yhat - y).pow(2)
        total_comp += diff2.sum(dim=(0, 1))
        
        for k, v in data_dict.items():
            accum_data[k] += v
        n_batches += 1
        
        pbar.set_postfix({'batch_loss': f'{loss.item():.4e}'})
    
    mean_loss = total_loss / max(total_elems, 1)
    num_positions = total_elems / 3.0 if total_elems > 0 else 1.0
    mean_comp = total_comp / num_positions
    
    loss_breakdown = {k: v / n_batches for k, v in accum_data.items()}
    
    return mean_loss, mean_comp.cpu(), loss_breakdown


# ============================================================================
# Main Fit Function
# ============================================================================

def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    t_shared: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    wd: float = 1e-6,
    device: str = "cpu",
    ckpt_path: str = "checkpoints/best_model.pt",
    metrics_path: Optional[str] = None,
    resume: bool = False,
    subwindow_length: Optional[int] = 256,
    lambda_physics: float = 0.1,
    lambda_bloch: float = 0.01,
    accumulation_steps: int = 1,
    warmup_epochs: int = 5,
    use_cosine_schedule: bool = True,
) -> None:
    """
    Enhanced training loop with:
    - Multi-scale loss
    - Physics regularization
    - Learning rate warmup and cosine annealing
    - Better checkpointing
    - Detailed logging
    """
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup losses
    data_loss_fn = MultiScaleLoss(
        w_full=1.0,
        w_checkpoints=0.5,
        w_signal=0.3,
        w_endpoint=0.2,
    ).to(device)
    
    physics_loss_fn = PhysicsLoss(
        lambda_fd=0.01,
        lambda_norm=1e-3,
        lambda_monotonic=1e-4,
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
        betas=(0.9, 0.999),
    )
    
    # Scheduler with warmup
    if use_cosine_schedule:
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=True,
        )
    
    # Mixed precision
    scaler = None
    if torch.cuda.is_available() and (
        (isinstance(device, torch.device) and device.type == "cuda") or
        (isinstance(device, str) and device.startswith("cuda"))
    ):
        scaler = GradScaler()
    
    # Resume from checkpoint
    best_val = float('inf')
    start_epoch = 0
    
    if resume and ckpt_path.exists():
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        best_val = ckpt.get('val', float('inf'))
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch}, best val: {best_val:.6e}")
    
    # CSV logging
    metrics_file = None
    if metrics_path:
        metrics_file = Path(metrics_path)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        if not metrics_file.exists():
            with metrics_file.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'train_loss', 'val_loss',
                    'train_Mx', 'train_My', 'train_Mz',
                    'val_Mx', 'val_My', 'val_Mz', 'lr'
                ])
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 100)
    
    for epoch in range(start_epoch, epochs):
        # Train
        tr_loss, tr_comp, tr_breakdown = train_one_epoch(
            model=model,
            loader=train_loader,
            t_shared=t_shared,
            optimizer=optimizer,
            data_loss_fn=data_loss_fn,
            physics_loss_fn=physics_loss_fn,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            scaler=scaler,
            subwindow_length=subwindow_length,
            lambda_physics=lambda_physics,
            lambda_bloch=lambda_bloch,
            accumulation_steps=accumulation_steps,
        )
        
        # Validate
        va_loss, va_comp, va_breakdown = evaluate(
            model=model,
            loader=val_loader,
            t_shared=t_shared,
            data_loss_fn=data_loss_fn,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
        )
        
        # Update scheduler
        if use_cosine_schedule:
            scheduler.step()
        else:
            scheduler.step(va_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging
        tr_Mx, tr_My, tr_Mz = tr_comp.tolist()
        va_Mx, va_My, va_Mz = va_comp.tolist()
        
        # Print epoch summary
        print(f"\n{'─'*100}")
        print(
            f"Epoch {epoch+1:3d}/{epochs} Summary: "
            f"Train={tr_loss:.6e} | Val={va_loss:.6e} | "
            f"ValErr[Mx={va_Mx:.3e}, My={va_My:.3e}, Mz={va_Mz:.3e}] | "
            f"LR={current_lr:.2e}"
        )
        print(f"{'─'*100}\n")
        
        # CSV logging
        if metrics_file:
            with metrics_file.open('a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, tr_loss, va_loss,
                    tr_Mx, tr_My, tr_Mz,
                    va_Mx, va_My, va_Mz,
                    current_lr
                ])
        
        # Save checkpoint
        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val': va_loss,
                'epoch': epoch,
            }, ckpt_path)
            print(f"✓ Saved best checkpoint (val={va_loss:.6e})\n")
    
    print("\n" + "=" * 100)
    print(f"Training complete! Best validation loss: {best_val:.6e}")
    print("=" * 100)




# ============================================================================
# CELL 1: Imports and Setup
# ============================================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


# ============================================================================
# CELL 2: Dataset Class
# ============================================================================

class BlochDataset(Dataset):
    """Dataset for Bloch simulation trajectories."""
    
    def __init__(self, data_path: str, device: str = "cpu"):
        """
        Load NPZ file containing:
        - t: (N, L) or (L,) time grid
        - y: (N, L, 3) magnetization trajectories
        - y0: (N, 3) initial states
        - u: (N, L, 4) controls [B1x, B1y, Gx, Gy]
        - p: (N, 5) parameters [T1, T2, dB0, B0, gamma]
        """
        data = np.load(data_path)
        
        self.t = torch.from_numpy(data['t']).float()
        self.y = torch.from_numpy(data['y']).float()
        self.y0 = torch.from_numpy(data['y0']).float()
        self.u = torch.from_numpy(data['u']).float()
        self.p = torch.from_numpy(data['p']).float()
        
        # If time is shared across all samples
        if self.t.ndim == 1:
            self.t_shared = True
        else:
            self.t_shared = False
        
        print(f"Loaded dataset: {len(self)} samples, {self.y.shape[1]} timesteps")
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        item = {
            'y': self.y[idx],
            'y0': self.y0[idx],
            'u': self.u[idx],
            'p': self.p[idx],
        }
        if not self.t_shared:
            item['t'] = self.t[idx]
        return item


# ============================================================================
# CELL 3: Create DataLoaders
# ============================================================================

# Paths to your data
data_dir = Path("data/npz_single_spin")  # or "data/npz" for ensemble

# Load datasets
train_dataset = BlochDataset(data_dir / "train.npz", device=device)
val_dataset = BlochDataset(data_dir / "val.npz", device=device)
test_dataset = BlochDataset(data_dir / "test.npz", device=device)

# Get shared time grid (assumes all samples use same time points)
t_shared = torch.from_numpy(np.load(data_dir / "train.npz")['t']).float().to(device)
print(f"Time grid shape: {t_shared.shape}, dt={t_shared[1]-t_shared[0]:.6f}")

# Create data loaders
# Note: Set num_workers=0 on Windows to avoid multiprocessing issues
# Set num_workers=4 on Linux/Mac for faster loading
batch_size = 32
num_workers = 0 if torch.cuda.is_available() else 0  # Use 0 on Windows, 4 on Linux

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")


# ============================================================================
# CELL 4: Instantiate Model
# ============================================================================

# Choose which model to use (import from your models file/cell)
# from improved_models import EnhancedBlochUDE, LatentBlochODE, AugmentedBlochODE

# Example: EnhancedBlochUDE
model = EnhancedBlochUDE(
    hidden_dim=256,
    num_layers=5,
    residual_scale=0.3,
    n_substeps=2,
    fourier_features=16,
    dropout=0.1,
).to(device)

# Count parameters
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {model.__class__.__name__}")
print(f"Trainable parameters: {n_params:,}")


# ============================================================================
# CELL 5: Quick Sanity Check (Forward Pass)
# ============================================================================

# Test forward pass
with torch.no_grad():
    sample_batch = next(iter(train_loader))
    y0_test = sample_batch['y0'].to(device)
    u_test = sample_batch['u'].to(device)
    p_test = sample_batch['p'].to(device)
    
    print(f"Input shapes:")
    print(f"  y0: {y0_test.shape}")
    print(f"  u: {u_test.shape}")
    print(f"  p: {p_test.shape}")
    print(f"  t: {t_shared.shape}")
    
    # Forward pass
    yhat_test = model(y0_test, t_shared, u_test, p_test)
    print(f"\nOutput shape: {yhat_test.shape}")
    print(f"Output range: [{yhat_test.min():.3f}, {yhat_test.max():.3f}]")
    print("✓ Forward pass successful!")


# ============================================================================
# CELL 6: Training Configuration
# ============================================================================

# Training hyperparameters
config = {
    'epochs': 100,
    'lr': 1e-3,
    'wd': 1e-6,  # Note: parameter is 'wd' not 'weight_decay'
    'subwindow_length': 256,  # For long sequences, train on random windows
    'lambda_physics': 0.1,    # Weight for physics losses
    'lambda_bloch': 0.01,     # Weight for Bloch RHS matching
    'accumulation_steps': 1,  # Gradient accumulation (increase if OOM)
    'warmup_epochs': 5,
    'use_cosine_schedule': True,
}

# Checkpoint paths
ckpt_dir = Path(f"checkpoints/{model.__class__.__name__}")
ckpt_dir.mkdir(parents=True, exist_ok=True)
ckpt_path = ckpt_dir / "best_model.pt"
metrics_path = ckpt_dir / "metrics.csv"

print("Configuration:")
for k, v in config.items():
    print(f"  {k}: {v}")


# ============================================================================
# CELL 7: Train the Model (Using Enhanced Loop)
# ============================================================================

# Import the enhanced training loop
# from improved_training_loop import fit

fit(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    t_shared=t_shared,
    device=device,
    ckpt_path=str(ckpt_path),
    metrics_path=str(metrics_path),
    resume=False,  # Set to True to resume from checkpoint
    **config
)


# ============================================================================
# CELL 8: Load Best Model and Evaluate
# ============================================================================

# Load best checkpoint
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model'])
print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
print(f"Best validation loss: {checkpoint['val']:.6e}")

# Evaluate on test set
model.eval()
test_losses = []
test_errors = {'Mx': [], 'My': [], 'Mz': []}

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        y = batch['y'].to(device)
        y0 = batch['y0'].to(device)
        u = batch['u'].to(device)
        p = batch['p'].to(device)
        
        yhat = model(y0, t_shared, u, p)
        
        # Component-wise errors
        errors = (yhat - y).pow(2).mean(dim=(0, 1))
        test_errors['Mx'].append(errors[0].cpu())
        test_errors['My'].append(errors[1].cpu())
        test_errors['Mz'].append(errors[2].cpu())
        
        # Overall loss
        loss = (yhat - y).pow(2).mean()
        test_losses.append(loss.cpu())

test_loss = torch.stack(test_losses).mean()
test_Mx = torch.stack(test_errors['Mx']).mean()
test_My = torch.stack(test_errors['My']).mean()
test_Mz = torch.stack(test_errors['Mz']).mean()

print(f"\nTest Results:")
print(f"  Overall MSE: {test_loss:.6e}")
print(f"  Mx MSE: {test_Mx:.6e}")
print(f"  My MSE: {test_My:.6e}")
print(f"  Mz MSE: {test_Mz:.6e}")


# ============================================================================
# CELL 9: Visualize Predictions
# ============================================================================

def plot_predictions(model, dataset, t_grid, n_samples=4, device='cuda'):
    """Plot predicted vs true trajectories."""
    model.eval()
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3.5*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            y_true = sample['y'].unsqueeze(0).to(device)
            y0 = sample['y0'].unsqueeze(0).to(device)
            u = sample['u'].unsqueeze(0).to(device)
            p = sample['p'].unsqueeze(0).to(device)
            
            y_pred = model(y0, t_grid, u, p)
            
            # Convert to numpy
            t_np = t_grid.cpu().numpy()
            y_true_np = y_true[0].cpu().numpy()
            y_pred_np = y_pred[0].cpu().numpy()
            
            # Compute errors
            errors = np.abs(y_true_np - y_pred_np)
            
            # Plot each component
            for j, (label, color_true, color_pred) in enumerate([
                ('Mx', 'steelblue', 'coral'),
                ('My', 'forestgreen', 'gold'),
                ('Mz', 'purple', 'pink')
            ]):
                ax = axes[i, j]
                
                # Plot true and predicted
                ax.plot(t_np, y_true_np[:, j], color=color_true, 
                       label='True', linewidth=2.5, alpha=0.8)
                ax.plot(t_np, y_pred_np[:, j], color=color_pred, 
                       label='Pred', linewidth=2, linestyle='--', alpha=0.9)
                
                # Shaded error region
                ax.fill_between(t_np, y_true_np[:, j] - errors[:, j], 
                               y_true_np[:, j] + errors[:, j],
                               alpha=0.2, color='red', label='Error')
                
                ax.set_xlabel('Time (s)', fontsize=11)
                ax.set_ylabel(label, fontsize=11)
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Add MSE text
                mse = np.mean((y_true_np[:, j] - y_pred_np[:, j])**2)
                ax.text(0.02, 0.98, f'MSE: {mse:.2e}', 
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                if i == 0:
                    ax.set_title(f'{label} Component', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(ckpt_dir / 'predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {ckpt_dir / 'predictions.png'}")

# Generate plots
plot_predictions(model, test_dataset, t_shared, n_samples=4, device=device)


# ============================================================================
# CELL 10: Plot Training Curves
# ============================================================================

import pandas as pd

def plot_training_curves(metrics_path):
    """Plot training and validation loss curves with detailed breakdown."""
    df = pd.read_csv(metrics_path)
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Main loss curves
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df['epoch'], df['train_loss'], label='Train', linewidth=2.5, color='steelblue')
    ax1.plot(df['epoch'], df['val_loss'], label='Val', linewidth=2.5, color='coral')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_yscale('log')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    
    # 2. Learning rate schedule
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(df['epoch'], df['lr'], linewidth=2, color='green')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Learning Rate', fontsize=11)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('LR Schedule', fontsize=12, fontweight='bold')
    
    # 3-5. Component errors over time
    for idx, (comp, color) in enumerate([('Mx', 'red'), ('My', 'blue'), ('Mz', 'green')]):
        ax = fig.add_subplot(gs[1, idx])
        ax.plot(df['epoch'], df[f'train_{comp}'], 
               label='Train', linewidth=2, alpha=0.7, color=color, linestyle='--')
        ax.plot(df['epoch'], df[f'val_{comp}'], 
               label='Val', linewidth=2.5, color=color)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(f'{comp} MSE', fontsize=11)
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{comp} Component Error', fontsize=12, fontweight='bold')
    
    # 6. Relative component contributions
    ax6 = fig.add_subplot(gs[2, 0])
    epochs = df['epoch'].values
    val_total = df['val_Mx'] + df['val_My'] + df['val_Mz']
    ax6.fill_between(epochs, 0, df['val_Mx']/val_total, label='Mx', alpha=0.6, color='red')
    ax6.fill_between(epochs, df['val_Mx']/val_total, 
                     (df['val_Mx'] + df['val_My'])/val_total, 
                     label='My', alpha=0.6, color='blue')
    ax6.fill_between(epochs, (df['val_Mx'] + df['val_My'])/val_total, 1, 
                     label='Mz', alpha=0.6, color='green')
    ax6.set_xlabel('Epoch', fontsize=11)
    ax6.set_ylabel('Relative Error Contribution', fontsize=11)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_title('Error Composition', fontsize=12, fontweight='bold')
    
    # 7. Training vs Validation gap
    ax7 = fig.add_subplot(gs[2, 1])
    gap = (df['val_loss'] - df['train_loss']) / df['train_loss'] * 100
    ax7.plot(epochs, gap, linewidth=2, color='purple')
    ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax7.set_xlabel('Epoch', fontsize=11)
    ax7.set_ylabel('Val-Train Gap (%)', fontsize=11)
    ax7.grid(True, alpha=0.3)
    ax7.set_title('Generalization Gap', fontsize=12, fontweight='bold')
    
    # 8. Improvement rate
    ax8 = fig.add_subplot(gs[2, 2])
    val_improvement = -np.diff(df['val_loss'].values) / df['val_loss'].values[:-1] * 100
    ax8.plot(epochs[1:], val_improvement, linewidth=2, color='orange')
    ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Epoch', fontsize=11)
    ax8.set_ylabel('Val Loss Improvement (%)', fontsize=11)
    ax8.grid(True, alpha=0.3)
    ax8.set_title('Training Progress', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Training Analysis - Best Val: {df["val_loss"].min():.6e}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(ckpt_dir / 'training_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {ckpt_dir / 'training_analysis.png'}")

plot_training_curves(metrics_path)


# ============================================================================
# CELL 11: Bloch Sphere Trajectory Visualization
# ============================================================================

def plot_bloch_sphere(model, dataset, t_grid, n_samples=2, device='cuda'):
    """
    Visualize magnetization trajectories on the Bloch sphere.
    Shows how M evolves in 3D space.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    model.eval()
    
    fig = plt.figure(figsize=(16, 8))
    
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    with torch.no_grad():
        for plot_idx, idx in enumerate(indices):
            sample = dataset[idx]
            y_true = sample['y'].unsqueeze(0).to(device)
            y0 = sample['y0'].unsqueeze(0).to(device)
            u = sample['u'].unsqueeze(0).to(device)
            p = sample['p'].unsqueeze(0).to(device)
            
            y_pred = model(y0, t_grid, u, p)
            
            # Convert to numpy
            y_true_np = y_true[0].cpu().numpy()
            y_pred_np = y_pred[0].cpu().numpy()
            
            # Create 3D plot
            ax = fig.add_subplot(1, n_samples, plot_idx + 1, projection='3d')
            
            # Plot unit sphere
            u_sphere = np.linspace(0, 2 * np.pi, 50)
            v_sphere = np.linspace(0, np.pi, 50)
            x_sphere = np.outer(np.cos(u_sphere), np.sin(v_sphere))
            y_sphere = np.outer(np.sin(u_sphere), np.sin(v_sphere))
            z_sphere = np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
            ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
            
            # Plot trajectories
            ax.plot(y_true_np[:, 0], y_true_np[:, 1], y_true_np[:, 2], 
                   'b-', linewidth=2.5, label='True', alpha=0.8)
            ax.plot(y_pred_np[:, 0], y_pred_np[:, 1], y_pred_np[:, 2], 
                   'r--', linewidth=2, label='Pred', alpha=0.8)
            
            # Mark start and end points
            ax.scatter(*y_true_np[0], color='green', s=100, marker='o', label='Start')
            ax.scatter(*y_true_np[-1], color='blue', s=100, marker='s', label='End (True)')
            ax.scatter(*y_pred_np[-1], color='red', s=100, marker='^', label='End (Pred)')
            
            # Axes
            ax.plot([0, 1.2], [0, 0], [0, 0], 'k-', linewidth=1, alpha=0.3)
            ax.plot([0, 0], [0, 1.2], [0, 0], 'k-', linewidth=1, alpha=0.3)
            ax.plot([0, 0], [0, 0], [0, 1.2], 'k-', linewidth=1, alpha=0.3)
            
            ax.set_xlabel('Mx', fontsize=11)
            ax.set_ylabel('My', fontsize=11)
            ax.set_zlabel('Mz', fontsize=11)
            ax.set_title(f'Sample {plot_idx + 1}: Bloch Sphere', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            
            # Equal aspect ratio
            ax.set_xlim([-1.2, 1.2])
            ax.set_ylim([-1.2, 1.2])
            ax.set_zlim([-1.2, 1.2])
    
    plt.tight_layout()
    plt.savefig(ckpt_dir / 'bloch_sphere.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {ckpt_dir / 'bloch_sphere.png'}")

# Generate Bloch sphere visualization
plot_bloch_sphere(model, test_dataset, t_shared, n_samples=2, device=device)


# ============================================================================
# CELL 12: Error Analysis by RF Pulse Type
# ============================================================================

def analyze_by_rf_type(model, dataset, t_grid, data_path, device='cuda'):
    """Analyze errors broken down by RF pulse type."""
    model.eval()
    
    # Load RF types from dataset
    data = np.load(data_path)
    rf_types = data['rf_type'] if 'rf_type' in data else None
    
    if rf_types is None:
        print("RF type information not available in dataset")
        return
    
    errors_by_type = {}
    component_errors = {}  # Track Mx, My, Mz separately
    
    print("Analyzing errors by RF pulse type...")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Processing"):
            sample = dataset[idx]
            rf_type = str(rf_types[idx])
            
            y_true = sample['y'].unsqueeze(0).to(device)
            y0 = sample['y0'].unsqueeze(0).to(device)
            u = sample['u'].unsqueeze(0).to(device)
            p = sample['p'].unsqueeze(0).to(device)
            
            y_pred = model(y0, t_grid, u, p)
            
            error = (y_pred - y_true).pow(2).mean().item()
            
            # Component-wise errors
            comp_err = (y_pred - y_true).pow(2).mean(dim=1)[0].cpu().numpy()
            
            if rf_type not in errors_by_type:
                errors_by_type[rf_type] = []
                component_errors[rf_type] = {'Mx': [], 'My': [], 'Mz': []}
            
            errors_by_type[rf_type].append(error)
            component_errors[rf_type]['Mx'].append(comp_err[0])
            component_errors[rf_type]['My'].append(comp_err[1])
            component_errors[rf_type]['Mz'].append(comp_err[2])
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overall error by RF type
    ax1 = axes[0, 0]
    rf_names = list(errors_by_type.keys())
    means = [np.mean(errors_by_type[rf]) for rf in rf_names]
    stds = [np.std(errors_by_type[rf]) for rf in rf_names]
    colors = plt.cm.Set3(np.linspace(0, 1, len(rf_names)))
    
    bars = ax1.bar(rf_names, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Mean Squared Error', fontsize=12)
    ax1.set_xlabel('RF Pulse Type', fontsize=12)
    ax1.set_title('Overall Error by RF Pulse Type', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2e}', ha='center', va='bottom', fontsize=9)
    
    # 2. Component-wise breakdown
    ax2 = axes[0, 1]
    x = np.arange(len(rf_names))
    width = 0.25
    
    mx_means = [np.mean(component_errors[rf]['Mx']) for rf in rf_names]
    my_means = [np.mean(component_errors[rf]['My']) for rf in rf_names]
    mz_means = [np.mean(component_errors[rf]['Mz']) for rf in rf_names]
    
    ax2.bar(x - width, mx_means, width, label='Mx', color='red', alpha=0.7)
    ax2.bar(x, my_means, width, label='My', color='blue', alpha=0.7)
    ax2.bar(x + width, mz_means, width, label='Mz', color='green', alpha=0.7)
    
    ax2.set_ylabel('Component MSE', fontsize=12)
    ax2.set_xlabel('RF Pulse Type', fontsize=12)
    ax2.set_title('Component-wise Errors by RF Type', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(rf_names, rotation=45, ha='right')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Error distribution (box plot)
    ax3 = axes[1, 0]
    data_for_box = [errors_by_type[rf] for rf in rf_names]
    bp = ax3.boxplot(data_for_box, labels=rf_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax3.set_ylabel('MSE', fontsize=12)
    ax3.set_xlabel('RF Pulse Type', fontsize=12)
    ax3.set_title('Error Distribution by RF Type', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_yscale('log')
    
    # 4. Sample count per RF type
    ax4 = axes[1, 1]
    counts = [len(errors_by_type[rf]) for rf in rf_names]
    ax4.bar(rf_names, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Number of Samples', fontsize=12)
    ax4.set_xlabel('RF Pulse Type', fontsize=12)
    ax4.set_title('Dataset Distribution', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add count labels
    for i, (name, count) in enumerate(zip(rf_names, counts)):
        ax4.text(i, count, str(count), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(ckpt_dir / 'rf_type_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {ckpt_dir / 'rf_type_analysis.png'}")
    
    # Print statistics
    print("\n" + "="*80)
    print("Error Statistics by RF Type:")
    print("="*80)
    for rf_type in rf_names:
        errors = errors_by_type[rf_type]
        mx_err = np.mean(component_errors[rf_type]['Mx'])
        my_err = np.mean(component_errors[rf_type]['My'])
        mz_err = np.mean(component_errors[rf_type]['Mz'])
        print(f"{rf_type:15s}: Overall={np.mean(errors):.6e}±{np.std(errors):.6e} | "
              f"Mx={mx_err:.3e} | My={my_err:.3e} | Mz={mz_err:.3e}")
    print("="*80 + "\n")

# Run analysis
test_data_path = data_dir / "test.npz"
analyze_by_rf_type(model, test_dataset, t_shared, test_data_path, device=device)


# ============================================================================
# CELL 13: Generate Full Report
# ============================================================================

def generate_report(model, test_dataset, t_shared, metrics_path, ckpt_dir, device='cuda'):
    """Generate a comprehensive performance report."""
    
    print("\n" + "="*100)
    print(" "*35 + "MODEL PERFORMANCE REPORT")
    print("="*100 + "\n")
    
    # Model info
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total Parameters: {n_params:,}")
    print(f"Checkpoint: {ckpt_dir.name}")
    print("-"*100)
    
    # Load training history
    df = pd.read_csv(metrics_path)
    
    # Best metrics
    best_epoch = df['val_loss'].idxmin()
    best_val = df['val_loss'].min()
    final_val = df['val_loss'].iloc[-1]
    
    print(f"\nTraining Summary:")
    print(f"  Total Epochs: {len(df)}")
    print(f"  Best Epoch: {best_epoch + 1}")
    print(f"  Best Val Loss: {best_val:.6e}")
    print(f"  Final Val Loss: {final_val:.6e}")
    print(f"  Improvement: {(1 - final_val/df['val_loss'].iloc[0])*100:.2f}%")
    
    # Test set evaluation
    model.eval()
    test_losses = []
    test_errors = {'Mx': [], 'My': [], 'Mz': []}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating test set", leave=False):
            y = batch['y'].to(device)
            y0 = batch['y0'].to(device)
            u = batch['u'].to(device)
            p = batch['p'].to(device)
            
            yhat = model(y0, t_shared, u, p)
            
            errors = (yhat - y).pow(2).mean(dim=(0, 1))
            test_errors['Mx'].append(errors[0].cpu())
            test_errors['My'].append(errors[1].cpu())
            test_errors['Mz'].append(errors[2].cpu())
            
            loss = (yhat - y).pow(2).mean()
            test_losses.append(loss.cpu())
    
    test_loss = torch.stack(test_losses).mean().item()
    test_Mx = torch.stack(test_errors['Mx']).mean().item()
    test_My = torch.stack(test_errors['My']).mean().item()
    test_Mz = torch.stack(test_errors['Mz']).mean().item()
    
    print(f"\nTest Set Performance:")
    print(f"  Overall MSE: {test_loss:.6e}")
    print(f"  Mx MSE: {test_Mx:.6e}")
    print(f"  My MSE: {test_My:.6e}")
    print(f"  Mz MSE: {test_Mz:.6e}")
    print(f"  RMSE: {np.sqrt(test_loss):.6e}")
    
    # Relative errors
    print(f"\nRelative Component Errors:")
    total_err = test_Mx + test_My + test_Mz
    print(f"  Mx: {test_Mx/total_err*100:.1f}%")
    print(f"  My: {test_My/total_err*100:.1f}%")
    print(f"  Mz: {test_Mz/total_err*100:.1f}%")
    
    # Comparison with validation
    val_loss_at_best = df['val_loss'].iloc[best_epoch]
    print(f"\nGeneralization:")
    print(f"  Val Loss (best): {val_loss_at_best:.6e}")
    print(f"  Test Loss: {test_loss:.6e}")
    print(f"  Gap: {(test_loss - val_loss_at_best)/val_loss_at_best*100:+.2f}%")
    
    print("\n" + "="*100)
    
    # Save report to file
    report_path = ckpt_dir / "performance_report.txt"
    with open(report_path, 'w') as f:
        f.write("MODEL PERFORMANCE REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {model.__class__.__name__}\n")
        f.write(f"Parameters: {n_params:,}\n")
        f.write(f"Best Val Loss: {best_val:.6e}\n")
        f.write(f"Test Loss: {test_loss:.6e}\n")
        f.write(f"Test RMSE: {np.sqrt(test_loss):.6e}\n\n")
        f.write("Component Errors:\n")
        f.write(f"  Mx: {test_Mx:.6e}\n")
        f.write(f"  My: {test_My:.6e}\n")
        f.write(f"  Mz: {test_Mz:.6e}\n")
    
    print(f"Report saved to: {report_path}\n")

# Generate report
generate_report(model, test_dataset, t_shared, metrics_path, ckpt_dir, device=device)