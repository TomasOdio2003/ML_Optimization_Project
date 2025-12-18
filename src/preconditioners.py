from __future__ import annotations
import torch
from dataclasses import dataclass
from .functional import residuals_from_theta_vec

@dataclass
class IdentityPreconditioner:
    def refresh(self, theta_vec: torch.Tensor, **kwargs):
        return

    def apply_inv(self, g: torch.Tensor) -> torch.Tensor:
        return g



@dataclass
class DiagGNPreconditioner:
    """
    M ≈ diag(J^T J) + damping*I

    """
    model: object
    template: object
    X: torch.Tensor
    y: torch.Tensor
    damping: float = 1e-4

    _diag: torch.Tensor | None = None
    _scale: torch.Tensor | None = None  # scalar normalization

    def refresh(self, theta_vec: torch.Tensor, J: torch.Tensor | None = None):
        if J is None:
            def r_of_v(v):
                return residuals_from_theta_vec(self.model, self.template, v, self.X, self.y)

            theta_vec = theta_vec.detach().requires_grad_(True)
            J = torch.autograd.functional.jacobian(r_of_v, theta_vec, create_graph=False)  # (m, n)

        diag = (J * J).sum(dim=0).detach()                 # diag(J^T J) >= 0
        diag = torch.clamp(diag, min=1e-12, max=1e12)      # keep sane
        diag = diag + self.damping

        # Robust scalar normalization (median is usually stable)
        scale = torch.median(diag)
        scale = torch.clamp(scale, min=1e-6, max=1e6)

        self._diag = diag
        self._scale = scale

    def apply_inv(self, g: torch.Tensor) -> torch.Tensor:
        assert self._diag is not None and self._scale is not None, "Call refresh() first."
        # Equivalent to using M_tilde = diag/scale so M_tilde^{-1} = scale/diag
        return (self._scale * g) / self._diag

