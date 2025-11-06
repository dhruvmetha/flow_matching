# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import math
from contextlib import nullcontext
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from flow_matching.solver.solver import Solver
from flow_matching.utils import ModelWrapper
from flow_matching.utils.manifolds import Manifold, geodesic

try:
    from tqdm import tqdm
    _TQDM = True
except Exception:
    _TQDM = False


# ---------- Low-level single-step methods on a manifold ----------

def _apply_velocity(
    manifold: Manifold,
    velocity_model: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t: Tensor,
    proju: bool,
) -> Tensor:
    u = velocity_model(x, t)
    return manifold.proju(x, u) if proju else u


def _project_x(manifold: Manifold, x: Tensor, projx: bool) -> Tensor:
    return manifold.projx(x) if projx else x


def _euler_step(
    manifold: Manifold,
    velocity_model: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t0: Tensor,
    dt: Tensor,
    *,
    projx: bool,
    proju: bool,
) -> Tensor:
    v = _apply_velocity(manifold, velocity_model, x, t0, proju)
    x = x + dt * v
    return _project_x(manifold, x, projx)


def _midpoint_step(
    manifold: Manifold,
    velocity_model: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t0: Tensor,
    dt: Tensor,
    *,
    projx: bool,
    proju: bool,
) -> Tensor:
    v0 = _apply_velocity(manifold, velocity_model, x, t0, proju)
    x_mid = _project_x(manifold, x + 0.5 * dt * v0, projx)
    v_mid = _apply_velocity(manifold, velocity_model, x_mid, t0 + 0.5 * dt, proju)
    x = x + dt * v_mid
    return _project_x(manifold, x, projx)


def _rk4_step(
    manifold: Manifold,
    velocity_model: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t0: Tensor,
    dt: Tensor,
    *,
    projx: bool,
    proju: bool,
) -> Tensor:
    k1 = _apply_velocity(manifold, velocity_model, x, t0, proju)
    k2 = _apply_velocity(manifold, velocity_model, _project_x(manifold, x + dt * k1 / 3, projx), t0 + dt / 3, proju)
    k3 = _apply_velocity(manifold, velocity_model, _project_x(manifold, x + dt * (k2 - k1 / 3), projx), t0 + 2 * dt / 3, proju)
    k4 = _apply_velocity(manifold, velocity_model, _project_x(manifold, x + dt * (k1 - k2 + k3), projx), t0 + dt, proju)
    x = x + (k1 + 3 * (k2 + k3) + k4) * dt * 0.125  # 1/8
    return _project_x(manifold, x, projx)


_STEP_FNS = {"euler": _euler_step, "midpoint": _midpoint_step, "rk4": _rk4_step}


# ---------- Utility for geodesic interpolation to arbitrary times ----------

def _geodesic_interp(
    manifold: Manifold,
    x0: Tensor,
    x1: Tensor,
    t0: Tensor,
    t1: Tensor,
    t_req: Tensor,
) -> Tensor:
    """Interpolate along the manifold geodesic between x0@t0 and x1@t1 at scalar time t_req."""
    # s in [0,1] (works for either direction since dt carries the sign)
    s = (t_req - t0) / (t1 - t0)
    # geodesic(manifold, a, b) -> callable gamma(s) that returns points on the manifold
    gamma = geodesic(manifold, x0, x1)
    return gamma(s)


# ---------- Public ODE integrator (shared by sampling & likelihood) ----------

def riemannian_odeint(
    *,
    manifold: Manifold,
    velocity_func: Callable[[Tensor, Tensor], Tensor],
    x_init: Tensor,
    time_grid: Tensor,
    step_size: Optional[float],
    method: str = "euler",
    projx: bool = True,
    proju: bool = True,
    return_intermediates: bool = False,
    verbose: bool = False,
    enable_grad: bool = False,
    accumulator_func: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
) -> Tuple[Union[Tensor, Sequence[Tensor]], Optional[Tensor]]:
    """
    Integrate a Riemannian ODE x' = u(x,t) defined by `velocity_func` on `manifold`.

    Args:
        manifold: Manifold instance.
        velocity_func: (x, t) -> velocity (same shape as x). May or may not be tangent; `proju` enforces tangency.
        x_init: initial state, shape (B, ...).
        time_grid: 1D tensor of times; **order is respected** (ascending or descending).
        step_size: if None, use consecutive pairs of `time_grid`; otherwise build a uniform discretization from
                   time_grid[0] -> time_grid[-1] with step ~ `step_size` (and include the exact endpoint).
        method: 'euler' | 'midpoint' | 'rk4'.
        projx, proju: project states and/or velocities onto manifold / tangent planes.
        return_intermediates: if True, return states at `time_grid` (with interpolation if uniform steps used).
        verbose: tqdm progress if available.
        enable_grad: set_grad_enabled around the whole integration (use True when training through the ODE).
        accumulator_func: optional integrand f(x,t) returning shape (B,), accumulated as ∑ f(x_t,t)*Δt.

    Returns:
        (solution, accumulator)
        solution:
            - if return_intermediates=False: final x_T, shape (B, ...).
            - if return_intermediates=True: stacked states at times in `time_grid`, shape (K, B, ...).
        accumulator:
            - None if no `accumulator_func` given; else shape (B,).
    """
    assert method in _STEP_FNS, f"Unknown method '{method}'."
    step_fn = _STEP_FNS[method]

    device = x_init.device
    time_grid = time_grid.to(device=device, dtype=torch.get_default_dtype())
    assert time_grid.ndim == 1 and time_grid.numel() >= 2, "time_grid must be a 1D tensor with at least 2 points."

    t0 = time_grid[0]
    tT = time_grid[-1]
    direction = 1.0 if (tT >= t0) else -1.0

    # Build discretization
    if step_size is None:
        t_disc = time_grid
    else:
        span = torch.abs(tT - t0).item()
        assert span > 0.0, "time_grid endpoints must be different when using step_size."
        n_steps = max(1, math.ceil(span / float(step_size)))
        # ensure exact endpoint inclusion, respect direction
        t_disc = torch.linspace(t0.item(), tT.item(), n_steps + 1, device=device)

    # Prepare accumulator
    acc = None
    if accumulator_func is not None:
        acc = torch.zeros(x_init.shape[0], device=device, dtype=x_init.dtype)

    # If we need to return intermediates, collect for the *requested* times.
    want_inter = return_intermediates
    if want_inter:
        # We'll fill these on the fly. Always include the initial state first.
        outs: list[Tensor] = []

    # Simple progress context
    if verbose:
        if not _TQDM:
            raise ImportError("tqdm is not installed but verbose=True was requested.")
        total = abs((tT - t0).item())
        pbar = tqdm(total=total if total > 0 else None, desc="NFE: 0")
        ctx = pbar
    else:
        ctx = nullcontext()

    x = x_init
    nfe = 0

    with ctx, torch.set_grad_enabled(enable_grad):
        # If we need exact samples at provided times and step_size is None, we can push x at each knot.
        if want_inter and step_size is None:
            outs.append(x.clone())

        # Iterate over consecutive pairs
        for a, b in zip(t_disc[:-1], t_disc[1:]):
            dt = b - a  # signed

            # Accumulate left-Riemann integrand if provided
            if accumulator_func is not None:
                acc = acc + accumulator_func(x, a) * dt

            # One step
            x_next = step_fn(manifold, velocity_func, x, a, dt, projx=projx, proju=proju)
            nfe += 1

            # Intermediates
            if want_inter:
                if step_size is None:
                    # Manual grid: exact knot — append x_next when we hit the next requested time
                    outs.append(x_next.clone())
                else:
                    # Uniform steps: append interpolated states for any requested t in [a,b] (either direction)
                    t_low, t_high = (a, b) if direction > 0 else (b, a)
                    # Find all requested times that fall into this segment (inclusive of the right endpoint only at the last step)
                    mask = (time_grid >= t_low) & (time_grid <= t_high)
                    # We may have multiple (e.g., coarse step_size, dense time_grid)
                    for t_req in time_grid[mask]:
                        # If t_req equals a boundary, prefer the boundary value to avoid tiny interpolation noise
                        if torch.isclose(t_req, a):
                            outs.append(x.clone())
                        elif torch.isclose(t_req, b):
                            outs.append(x_next.clone())
                        else:
                            outs.append(_geodesic_interp(manifold, x, x_next, a, b, t_req))

            # Advance
            x = x_next

            if verbose and _TQDM:
                ctx.n = abs((a - t0).item())
                ctx.set_description(f"NFE: {nfe}")
                ctx.refresh()

    if want_inter:
        # `outs` should have length == len(time_grid)
        sol = torch.stack(outs, dim=0)
    else:
        sol = x

    return sol, acc


# ---------- Riemannian ODE solver class ----------

class RiemannianODESolver(Solver):
    """Riemannian ODE solver wrapping a velocity model and a manifold."""

    def __init__(self, manifold: Manifold, velocity_model: ModelWrapper):
        super().__init__()
        self.manifold = manifold
        self.velocity_model = velocity_model

    # -------- Public sampling (now just a thin wrapper around riemannian_odeint) --------
    def sample(
        self,
        x_init: Tensor,
        step_size: Optional[float],
        projx: bool = True,
        proju: bool = True,
        method: str = "euler",
        time_grid: Optional[Tensor] = None,
        return_intermediates: bool = False,
        verbose: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Union[Tensor, Sequence[Tensor]]:
        """
        Solve forward from min(time_grid) -> max(time_grid). If not provided, uses [0, 1].
        Returns the final state unless `return_intermediates=True` (then returns states at `time_grid`).
        """
        if time_grid is None:
            time_grid = torch.tensor([0.0, 1.0], device=x_init.device, dtype=torch.get_default_dtype())
        else:
            time_grid = time_grid.to(device=x_init.device, dtype=torch.get_default_dtype())
        # Preserve original "forward" semantics: integrate on an increasing grid.
        time_grid = torch.sort(time_grid).values

        def velocity_func(x: Tensor, t: Tensor) -> Tensor:
            return self.velocity_model(x=x, t=t, **model_extras)

        sol, _ = riemannian_odeint(
            manifold=self.manifold,
            velocity_func=velocity_func,
            x_init=x_init,
            time_grid=time_grid,
            step_size=step_size,
            method=method,
            projx=projx,
            proju=proju,
            return_intermediates=return_intermediates,
            verbose=verbose,
            enable_grad=enable_grad,
            accumulator_func=None,
        )
        return sol

    # -------- Likelihood via reverse-time integration (CNF-style) --------
    def compute_likelihood(
        self,
        x_1: Tensor,
        log_p0: Callable[[Tensor], Tensor],
        step_size: Optional[float],
        method: str = "euler",
        time_grid: Optional[Tensor] = None,
        return_intermediates: bool = False,
        *,
        exact_divergence: bool = False,
        hutchinson: str = "rademacher",  # 'rademacher' or 'gaussian'
        projx: bool = True,
        proju: bool = True,
        verbose: bool = False,
        # Critical gradient controls:
        retain_path_grad: bool = False,           # True for training: keep ∂x/∂θ through the trajectory
        create_graph_for_divergence: bool = False,  # True for training: allow grad-of-grad to flow to θ
        detach_divergence: bool = False,          # True for eval: cut grads on the integrand only
        **model_extras,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]:
        """
        Compute log p(x_1) by integrating from t=1 → t=0:
            log p(x_1) = log p0(x_0) + ∫_{1}^{0} div u(x_t, t) dt

        Args worth noting:
            exact_divergence: if False, uses Hutchinson estimator with a single probe vector `z` fixed across time.
            retain_path_grad: keep trajectory grads (no detach of x) so ∂x/∂θ contributes — use for training.
            create_graph_for_divergence: build a higher-order graph for divergence so grads flow through autograd.grad.
            detach_divergence: detach only the divergence integrand (cheap evaluation).

        Returns:
            If return_intermediates=False: (x0, log_prob) with shapes (B, ...), (B,)
            If return_intermediates=True: (traj, log_prob) with shapes (K, B, ...), (B,)
        """
        device = x_1.device
        if time_grid is None:
            time_grid = torch.tensor([1.0, 0.0], device=device, dtype=torch.get_default_dtype())
        else:
            time_grid = time_grid.to(device=device, dtype=torch.get_default_dtype())
        assert time_grid[0] > time_grid[-1], "Likelihood integration expects a reverse-time grid starting at 1.0 and ending at 0.0."

        # Fix one Hutchinson probe z across all steps to reduce variance (standard practice)
        if not exact_divergence:
            if hutchinson == "rademacher":
                z = torch.empty_like(x_1).bernoulli_(0.5).mul_(2.0).sub_(1.0)  # ±1
            elif hutchinson == "gaussian":
                z = torch.randn_like(x_1)
            else:
                raise ValueError("hutchinson must be 'rademacher' or 'gaussian'")
        else:
            z = None

        def velocity_func(x: Tensor, t: Tensor) -> Tensor:
            return self.velocity_model(x=x, t=t, **model_extras)

        # Divergence integrand f(x,t): (B,)
        def divergence_func(x: Tensor, t: Tensor) -> Tensor:
            """
            Computes div u(x,t) w.r.t. ambient coordinates.
            - If retain_path_grad is False, we detach `x` here → cheaper but drops ∂x/∂θ pathway.
            - If create_graph_for_divergence is True, we allow higher-order grads for training.
            """
            x_div = x if retain_path_grad else x.detach()
            x_div = x_div.requires_grad_(True)

            u = velocity_func(x_div, t)
            if proju:
                u = self.manifold.proju(x_div, u)

            # Flatten to (B, D)
            u_flat = u.reshape(u.shape[0], -1)
            if exact_divergence:
                # Exact trace: sum_j ∂u_j/∂x_j per sample (O(D) vjp's)
                div = torch.zeros(x_div.shape[0], device=x_div.device, dtype=x_div.dtype)
                for j in range(u_flat.shape[1]):
                    g = torch.autograd.grad(
                        u_flat[:, j].sum(), x_div,
                        create_graph=create_graph_for_divergence,
                        retain_graph=create_graph_for_divergence,
                        allow_unused=False,
                    )[0]
                    div = div + g.reshape(g.shape[0], -1)[:, j]
            else:
                # Hutchinson: E_z[z^T J_u(x) z] using a single probe fixed across time
                z_flat = z.reshape(z.shape[0], -1)
                u_dot_z = (u_flat * z_flat).sum(dim=1)  # (B,)
                grad_u_dot_z = torch.autograd.grad(
                    u_dot_z, x_div,
                    create_graph=create_graph_for_divergence,
                    retain_graph=create_graph_for_divergence,
                    allow_unused=False,
                )[0]
                div = (grad_u_dot_z.reshape(grad_u_dot_z.shape[0], -1) * z_flat).sum(dim=1)

            if detach_divergence:
                div = div.detach()
            return div  # shape (B,)

        # Integrate 1 → 0 (dt is negative automatically)
        sol, log_det = riemannian_odeint(
            manifold=self.manifold,
            velocity_func=velocity_func,
            x_init=x_1,
            time_grid=time_grid,
            step_size=step_size,
            method=method,
            projx=projx,
            proju=proju,
            return_intermediates=return_intermediates,
            verbose=verbose,
            enable_grad=(retain_path_grad or create_graph_for_divergence),  # build graph only when needed
            accumulator_func=divergence_func,
        )

        x0 = sol[-1] if return_intermediates else sol
        log_p0_x0 = log_p0(x0)
        log_p_x1 = log_p0_x0 + log_det  # dt sign already handled by the integrator

        if return_intermediates:
            return sol, log_p_x1
        else:
            return x0, log_p_x1