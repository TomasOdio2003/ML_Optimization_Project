from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple
from typing import Callable, Tuple, Optional
from .functional import residuals_from_theta_vec

def armijo_backtracking(theta, f, g, p, obj_fn, alpha0=1.0, c=1e-4, tau=0.5, max_ls=30):
    # Armijo: f(theta + a p) <= f + c a g^T p
    gTp = float(torch.dot(g, p))
    alpha = alpha0
    for _ in range(max_ls):
        f_new, _ = obj_fn(theta + alpha * p)
        if float(f_new) <= float(f) + c * alpha * gTp:
            return alpha
        alpha *= tau
    return alpha


def armijo_backtracking_count(theta, f, g, p, obj_fn, alpha0=1.0, c=1e-4, tau=0.5, max_ls=30):
    gTp = float(torch.dot(g, p))
    alpha = alpha0
    nfev = 0

    f0 = float(f)
    # numerical tolerance to avoid "can't see decrease" stalls
    tol = 1e-12 * (abs(f0) + 1.0)

    for _ in range(max_ls):
        f_new, _ = obj_fn(theta + alpha * p)
        nfev += 1
        f_new = float(f_new)

        if f_new <= f0 + c * alpha * gTp + tol:
            return alpha, nfev

        alpha *= tau

    return alpha, nfev


def _zoom_strong_wolfe(theta, p, obj_fn, phi0, derphi0, a_lo, a_hi, phi_lo, c1, c2, max_zoom=25):
    """
    Zoom phase from Nocedal–Wright strong Wolfe line search.
    Returns (alpha_star, nfev).
    """
    nfev = 0
    for _ in range(max_zoom):
        a = 0.5 * (a_lo + a_hi)  # bisection; simple + robust

        f_a, g_a = obj_fn(theta + a * p)
        nfev += 1
        phi_a = float(f_a)

        if (phi_a > phi0 + c1 * a * derphi0) or (phi_a >= phi_lo):
            a_hi = a
        else:
            derphi_a = float(torch.dot(g_a, p))
            if abs(derphi_a) <= -c2 * derphi0:
                return a, nfev

            if derphi_a * (a_hi - a_lo) >= 0:
                a_hi = a_lo

            a_lo = a
            phi_lo = phi_a

        if abs(a_hi - a_lo) < 1e-12:
            return a, nfev

    return a, nfev


def strong_wolfe_line_search(
    theta, f0, g0, p, obj_fn,
    alpha0=1.0, c1=1e-4, c2=0.9,
    alpha_max=50.0, max_iters=25, max_zoom=25
):
    """
    Strong Wolfe line search (Nocedal–Wright style).
    Returns (alpha, nfev) where nfev counts obj_fn evaluations.
      Wolfe:
        f(x+a p) <= f(x) + c1 a g^T p
        |g(x+a p)^T p| <= c2 |g^T p|
    """
    phi0 = float(f0)
    derphi0 = float(torch.dot(g0, p))
    nfev = 0

    # Ensure descent direction; if not, fall back to steepest descent
    if derphi0 >= 0.0:
        p = -g0
        derphi0 = float(torch.dot(g0, p))

    a_prev = 0.0
    phi_prev = phi0
    a = alpha0

    for i in range(max_iters):
        f_a, g_a = obj_fn(theta + a * p)
        nfev += 1
        phi_a = float(f_a)

        if (phi_a > phi0 + c1 * a * derphi0) or (i > 0 and phi_a >= phi_prev):
            a_star, nzoom = _zoom_strong_wolfe(theta, p, obj_fn, phi0, derphi0, a_prev, a, phi_prev, c1, c2, max_zoom=max_zoom)
            return a_star, nfev + nzoom

        derphi_a = float(torch.dot(g_a, p))
        if abs(derphi_a) <= -c2 * derphi0:
            return a, nfev

        if derphi_a >= 0.0:
            a_star, nzoom = _zoom_strong_wolfe(theta, p, obj_fn, phi0, derphi0, a, a_prev, phi_a, c1, c2, max_zoom=max_zoom)
            return a_star, nfev + nzoom

        a_prev = a
        phi_prev = phi_a
        a = min(2.0 * a, alpha_max)

    return a, nfev


def gradient_descent(theta0, obj_fn, max_iters, tol_grad, alpha0=1.0, use_line_search=True, callback=None):
    theta = theta0.clone().detach()
    for k in range(max_iters):
        f, g = obj_fn(theta)
        gnorm = float(torch.linalg.norm(g))

        if callback:
            callback(k, theta, f, g, alpha=None)

        if gnorm <= tol_grad:
            break

        p = -g
        alpha, ls_evals = armijo_backtracking_count(theta, f, g, p, obj_fn, alpha0=alpha0) if use_line_search else (alpha0, 0)
        theta = theta + alpha * p
        if callback:
            callback(k, theta, f, g, alpha=alpha, ls_evals=ls_evals)


    return theta

def preconditioned_gd(theta0, obj_fn, precond, max_iters, tol_grad, alpha0=1.0, use_line_search=True,
                      precond_refresh=10, callback=None):
    theta = theta0.clone().detach()

    for k in range(max_iters):
        f, g = obj_fn(theta)
        gnorm = float(torch.linalg.norm(g))

        if k % precond_refresh == 0:
            precond.refresh(theta)

        if callback:
            callback(k, theta, f, g, alpha=None)

        if gnorm <= tol_grad:
            break

        # Proposed preconditioned direction
        p = -precond.apply_inv(g)

        # (A) Ensure descent direction; otherwise fall back to steepest descent
        gTp = float(torch.dot(g, p))
        if (gTp >= 0.0) or (not torch.isfinite(torch.tensor(gTp))):
            p = -g
            gTp = float(torch.dot(g, p))

        # Line search
        alpha, ls_evals = armijo_backtracking_count(theta, f, g, p, obj_fn, alpha0=alpha0) if use_line_search else (alpha0, 0)

        # (B) If line search collapses, recover with steepest descent once
        if alpha < 1e-12:
            p = -g
            alpha, ls_evals = armijo_backtracking_count(theta, f, g, p, obj_fn, alpha0=alpha0) if use_line_search else (alpha0, 0)

        theta = (theta + alpha * p).detach()

        # Correct logging: log f/g at the new theta
        if callback:
            f_new, g_new = obj_fn(theta)
            callback(k, theta, f_new, g_new, alpha=alpha, ls_evals=ls_evals, gTp=gTp, p_norm=float(torch.linalg.norm(p)))

    return theta



def inv_diag_bfgs_update_16(v_diag: torch.Tensor, s: torch.Tensor, y: torch.Tensor, damping: float = 1e-8) -> torch.Tensor:
    """
    Diagonal inverse quasi-Newton update (paper eq. (16)):

      v_{k+1,i} =
        (1 - (y_i s_i)/(y^T s))^2 * v_{k,i}
        + (2 - (s_i y_i)/(y^T s)) * (s_i^2)/(y^T s)

    where v_diag approximates diag( (∇^2 f)^{-1} ).
    """
    ys = torch.dot(y, s)
    if float(ys) <= 1e-12:
        # Curvature condition failed; keep v as-is
        return v_diag

    t = (s * y) / ys  # elementwise (s_i y_i)/(y^T s)
    v_new = (1.0 - t) ** 2 * v_diag + (2.0 - t) * (s * s) / ys

    # Keep strictly positive for stability
    v_new = torch.clamp(v_new, min=damping)
    return v_new



def damp_curvature_pair_y(s: torch.Tensor, y: torch.Tensor, delta: float = 1e-4) -> torch.Tensor:
    """
    Ensure a minimum curvature: y^T s >= delta * ||s||^2
    by replacing y with y + lambda*s when needed.

    This is a simple curvature damping / regularization that helps in nonconvex/noisy settings.
    """
    ys = float(torch.dot(y, s))
    sTs = float(torch.dot(s, s)) + 1e-30  # avoid divide-by-zero

    # If curvature is too small/nonpositive, add lambda*s to y
    if ys < delta * sTs:
        lam = (delta * sTs - ys) / sTs
        y = y + lam * s

    return y

def phi_inv_from_xi_alpha(t: float, xi: float, alpha: float) -> float:
    # phi(t) = xi * t^alpha  =>  phi^{-1}(u) = (u/xi)^(1/alpha)
    t = max(float(t), 0.0)
    xi = float(xi)
    alpha = float(alpha)
    if xi <= 0.0 or alpha <= 0.0:
        raise ValueError("Need xi>0 and alpha>0 for phi(t)=xi*t^alpha.")
    return (t / xi) ** (1.0 / alpha)

def inv_diag_bfgs_update_25(
    v_diag: torch.Tensor,
    s: torch.Tensor,
    y: torch.Tensor,
    gnorm_xk: float,          # ||∇f(x_k)|| at the OLD iterate
    eps: float = 0.2,         # ε in (0,1)
    xi: float = 100.0,        # φ(t)=xi*t^alpha  (bigger xi => smaller φ^{-1})
    alpha: float = 1.0,
    damping: float = 1e-8,
) -> torch.Tensor:
    """
    Paper eq. (25):
      let t_i = (y_i s_i)/(y^T s), threshold = eps * min(1, ||∇f(x_k)||)

      if t_i < threshold:
         (V_{k+1})_i = (V_k)_i
      else:
         (V_{k+1})_i =
            (1 - t_i)^2 (V_k)_i
            + max(0, 2 - t_i) * (s_i^2)/(y^T s)
            + φ^{-1}(||∇f(x_k)||)

    Requires y^T s > 0; otherwise returns v_diag unchanged.
    """
    ys = torch.dot(y, s)
    if float(ys) <= 1e-12 or (not torch.isfinite(ys)):
        return v_diag

    t = (s * y) / ys  # elementwise t_i
    thresh = float(eps) * min(1.0, float(gnorm_xk))

    add = phi_inv_from_xi_alpha(float(gnorm_xk), xi=xi, alpha=alpha)

    v_new = v_diag.clone()
    mask = t >= thresh
    if torch.any(mask):
        core = (1.0 - t) ** 2 * v_diag + torch.clamp(2.0 - t, min=0.0) * (s * s) / ys + add
        v_new[mask] = core[mask]

    v_new = torch.clamp(v_new, min=damping)
    return v_new


@dataclass
class LBFGSState:
    s_list: List[torch.Tensor]
    y_list: List[torch.Tensor]
    rho_list: List[float]
    gamma: float
    m_hist: int
    # New: diagonal initial inverse-Hessian approx for L-BFGS-D
    H0_diag: Optional[torch.Tensor] = None  # shape (n,)

def _two_loop_recursion(g: torch.Tensor, state: LBFGSState) -> torch.Tensor:
    q = g.clone()
    alpha = []

    for s, y, rho in zip(reversed(state.s_list), reversed(state.y_list), reversed(state.rho_list)):
        a = rho * float(torch.dot(s, q))
        alpha.append(a)
        q = q - a * y

    if state.H0_diag is not None:
        r = state.H0_diag * q          
    else:
        r = state.gamma * q            

    for (s, y, rho), a in zip(zip(state.s_list, state.y_list, state.rho_list), reversed(alpha)):
        b = rho * float(torch.dot(y, r))
        r = r + s * (a - b)

    return r


def lbfgs_minimize(
    theta0: torch.Tensor,
    obj_fn,
    max_iters: int,
    tol_grad: float,
    history_size: int = 10,
    alpha0: float = 1.0,
    use_line_search: bool = True,
    callback=None,
    line_search: str = "armijo",   # "armijo" or "strong_wolfe",
) -> Tuple[torch.Tensor, LBFGSState]:
    """
    L-BFGS optimizer on the flat parameter vector theta.
    Returns (theta_final, lbfgs_state) where lbfgs_state contains the inverse-Hessian approximation info.
    """
    theta = theta0.clone().detach()
    f, g = obj_fn(theta)
    gnorm = float(torch.linalg.norm(g))

    state = LBFGSState(s_list=[], y_list=[], rho_list=[], gamma=1.0, m_hist=history_size)

    if callback:
        callback(0, theta, f, g, alpha=None)

    if gnorm <= tol_grad:
        return theta, state

    for k in range(1, max_iters + 1):
        # direction p = -H_k g
        if len(state.s_list) == 0:
            p = -g
        else:
            p = -_two_loop_recursion(g, state)

        p_norm = float(torch.linalg.norm(p))

        if use_line_search:
            if line_search == "strong_wolfe":
                alpha, ls_evals = strong_wolfe_line_search(theta, f, g, p, obj_fn, alpha0=alpha0)
            else:
                alpha, ls_evals = armijo_backtracking_count(theta, f, g, p, obj_fn, alpha0=alpha0)
        else:
            alpha, ls_evals = alpha0, 0

        step_norm = float(torch.linalg.norm(alpha * p))
        
        theta_new = (theta + alpha * p).detach()

        f_new, g_new = obj_fn(theta_new)
        s = (theta_new - theta).detach()
        y = (g_new - g).detach()

        ys = float(torch.dot(y, s))
        yy = float(torch.dot(y, y))

        if ys > 1e-12 and yy > 1e-12:
            rho = 1.0 / ys
            state.s_list.append(s)
            state.y_list.append(y)
            state.rho_list.append(rho)

            if len(state.s_list) > state.m_hist:
                state.s_list.pop(0)
                state.y_list.pop(0)
                state.rho_list.pop(0)

            state.gamma = ys / yy

        theta, f, g = theta_new, f_new.detach(), g_new.detach()

        if callback:
            try:
                callback(k, theta, f, g, alpha=alpha,
                         ls_evals=ls_evals,
                         p_norm=p_norm,
                         step_norm=step_norm,
                         gamma=state.gamma)
            except TypeError:
                callback(k, theta, f, g, alpha=alpha)


        if float(torch.linalg.norm(g)) <= tol_grad:
            break

    return theta, state


def lbfgs_d_minimize(
    theta0: torch.Tensor,
    obj_fn,
    max_iters: int,
    tol_grad: float,
    history_size: int = 10,
    alpha0: float = 1.0,
    use_line_search: bool = True,
    diag_damping: float = 1e-8,
    callback=None,
    line_search: str = "armijo",
    update_eps: float = 0.10,      # ε in (0,1)
    phi_xi: float = 1e6,        # ξ
    phi_alpha: float = 1.0,       # α
) -> Tuple[torch.Tensor, LBFGSState]:
    """
    Diagonal L-BFGS (L-BFGS-D): same two-loop recursion, but H0_k is diagonal
    and updated using eq. (16). That diagonal is used as the initial matrix in the recursion.
    """
    theta = theta0.clone().detach()
    f, g = obj_fn(theta)


    # Start with positive diagonal H0
    v_diag = torch.ones_like(theta)  # H0_diag
    state = LBFGSState(
        s_list=[], y_list=[], rho_list=[],
        gamma=1.0, m_hist=history_size,
        H0_diag=v_diag
    )

    if callback:
        callback(0, theta, f, g, alpha=None)

    if float(torch.linalg.norm(g)) <= tol_grad:
        return theta, state

    for k in range(1, max_iters + 1):
        # p = -H_k g
        if len(state.s_list) == 0:
            p = -(state.H0_diag * g) 
        else:
            p = -_two_loop_recursion(g, state)
        
        if float(torch.dot(g, p)) >= 0.0:
            p = -g
        p_norm = float(torch.linalg.norm(p))

        if use_line_search:
            if line_search == "strong_wolfe":
                alpha, ls_evals = strong_wolfe_line_search(theta, f, g, p, obj_fn, alpha0=alpha0)
            else:
                alpha, ls_evals = armijo_backtracking_count(theta, f, g, p, obj_fn, alpha0=alpha0)
        else:
            alpha, ls_evals = alpha0, 0

        step_norm = float(torch.linalg.norm(alpha * p))    

        theta_new = (theta + alpha * p).detach()

        f_new, g_new = obj_fn(theta_new)
        s = (theta_new - theta).detach()
        y = (g_new - g).detach()

        ys = float(torch.dot(y, s))
        yy = float(torch.dot(y, y))

        if ys > 1e-12 and yy > 1e-12:
            rho = 1.0 / ys

            state.s_list.append(s)
            state.y_list.append(y)
            state.rho_list.append(rho)

            if len(state.s_list) > state.m_hist:
                state.s_list.pop(0)
                state.y_list.pop(0)
                state.rho_list.pop(0)

            # Update diagonal initial matrix using eq. (25)
            gnorm_xk = float(torch.linalg.norm(g))  # g at x_k (OLD)
            v_diag = inv_diag_bfgs_update_25(
                v_diag=v_diag,
                s=s,
                y=y,
                gnorm_xk=gnorm_xk,
                eps=update_eps,
                xi=phi_xi,
                alpha=phi_alpha,
                damping=diag_damping,
            )
            state.H0_diag = v_diag


        theta, f, g = theta_new, f_new.detach(), g_new.detach()

        if callback:
            try:
                h0 = state.H0_diag
                callback(k, theta, f, g, alpha=alpha,
                         ls_evals=ls_evals,
                         p_norm=p_norm,
                         step_norm=step_norm,
                         h0_min=float(h0.min()),
                         h0_mean=float(h0.mean()),
                         h0_max=float(h0.max()))
            except TypeError:
                callback(k, theta, f, g, alpha=alpha)

        if float(torch.linalg.norm(g)) <= tol_grad:
            break

    return theta, state




