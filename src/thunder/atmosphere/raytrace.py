"""Ray acoustics in a horizontally stratified, moving atmosphere (SPEC.md Phase 3).

Physics. With dispersion relation  c(z)|s| + v(z).s = 1  (s = wave slowness, v = horizontal
wind), the Hamiltonian ray equations are
    dx/dt = c s/|s| + v,      ds_z/dt = -(|s| c'(z) + s_h . v'(z)),      ds_h/dt = 0.
The horizontal slowness s_h is conserved (Snell's law), so |s|(z) = (1 - v(z).s_h) / c(z) and
s_z(z)^2 = S(z) = |s|^2 - |s_h|^2. Along a ray monotonic in z, with dz/dt = c |s_z| / |s|:
    T = int |s| / (c |s_z|) dz,      X = int (s_h + |s| v / c) / |s_z| dz
(X = horizontal displacement from source to receiver), and the Jacobian dX/ds_h is
    J = int (I - v v^T / c^2) / |s_z| dz + int (c s_h + |s| v)(|s| v / c + s_h)^T / (c |s_z|^3) dz.
Each layer between grid heights is integrated assuming S linear in z, which integrates the
1/|s_z| and 1/|s_z|^3 factors exactly (so rays near their turning height stay accurate):
    int dz / sqrt(S) = 2 dz / (r0 + r1),     int dz / S^(3/2) = 2 dz / (r0 r1 (r0 + r1)),  r = sqrt(S),
with the smooth numerators averaged over the layer (second-order accurate in dz).

Eigenrays (the ray from a given source to a given receiver) are found by a bracketed Newton
iteration on |s_h| along the source-to-receiver azimuth, then a 2D Newton step on s_h for the
sideways drift that crosswind causes. A target beyond the limiting ray (s_z -> 0 somewhere on
the path) is in the acoustic shadow: no direct ray.

Spreading: energy flux along a ray tube with impedance rho c (the O(Mach) moving-medium terms
of the Blokhintzev invariant are neglected):
    A^2 = (rho_r c_r) / (rho_s c_s) * dOmega / dA_perp,
    dOmega = |det(dn_h/ds_h)| / |n_z,s| d^2s_h,    dA_perp = |det J| |g_z,r| d^2s_h,
with n the unit wave normal and g the group-velocity direction. In uniform still air this is
exactly 1/R^2 (tested).

`trace_ray` integrates the Hamiltonian equations with scipy.integrate.solve_ivp; it is the
independent reference the quadrature solver is tested against, and draws ray fans.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from thunder.types import FloatArray, FloatLike

_CHUNK = 1024
_EPS_SIGMA = 1e-10


GROUND_STEP_M = 0.05  # first layer thickness of the height grid
GROWTH = 1.15  # each layer is at most 15% thicker than the one below


def height_nodes(z_top: float, dz_max: float, breakpoints: tuple[float, ...] = ()) -> FloatArray:
    """Height grid: 5 cm at the ground, thickness growing by 15% per layer up to dz_max.

    Layer thickness ~ 0.15 z suits the near-ground wind profile (a power law, scale-free in
    z), and coarser layers aloft are accurate because the 1/s_z factors are integrated
    exactly per layer and the remaining numerators vary smoothly (docs/log.md: convergence).
    """
    z = [0.0]
    step = GROUND_STEP_M
    while z[-1] < z_top:
        z.append(min(z[-1] + step, z_top))
        step = min(step * GROWTH, dz_max)
    # Profile kinks (inversion edges, tropopause) become nodes so no layer straddles one.
    extra = [b for b in breakpoints if 0.0 < b < z_top]
    nodes = np.unique(np.concatenate([np.array(z), extra]))
    return nodes[np.concatenate([[True], np.diff(nodes) > 1e-6])]


@dataclass(frozen=True)
class RayGrid:
    """Profiles tabulated on a height grid (graded near the ground) for the ray integrals."""

    z: FloatArray  # (n,) node heights
    c: FloatArray  # (n,)
    v: FloatArray  # (n, 2) horizontal wind
    rho: FloatArray  # (n,)

    @classmethod
    def from_atmosphere(cls, atm, z_top: float, dz: float, breakpoints: tuple[float, ...] = ()) -> RayGrid:
        z = height_nodes(z_top, dz, breakpoints)
        return cls(z, atm.sound_speed(z), atm.wind(z)[:, :2], atm.density(z))

    def first_node_at_or_above(self, zq: FloatLike) -> np.ndarray:
        return np.searchsorted(self.z, np.asarray(zq) - 1e-9, side="left")

    def last_node_at_or_below(self, zq: FloatLike) -> np.ndarray:
        return np.searchsorted(self.z, np.asarray(zq) + 1e-9, side="right") - 1

    def at(self, zq: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Linearly interpolated c, v (..., 2), rho at heights zq."""
        zq = np.clip(np.asarray(zq, dtype=float), self.z[0], self.z[-1])
        c = np.interp(zq, self.z, self.c)
        v = np.stack([np.interp(zq, self.z, self.v[:, 0]), np.interp(zq, self.z, self.v[:, 1])], axis=-1)
        return c, v, np.interp(zq, self.z, self.rho)


def _point_terms(sh: FloatArray, c: FloatArray, v: FloatArray):
    """|s|, S = s_z^2, and the integrand numerators at points (broadcast over leading axes)."""
    sabs = (1.0 - np.sum(sh * v, axis=-1)) / c
    S = sabs**2 - np.sum(sh * sh, axis=-1)
    a = sabs / c  # time numerator
    b = sh + (sabs / c)[..., None] * v  # displacement numerator (2)
    return sabs, S, a, b


def layer_weights(r0, r1, dz):
    """Exact weights of the two layer nodes for int N / sqrt(S) and int N / S^(3/2) over a layer
    where both the numerator N and S = r^2 vary linearly:
        int (N0 (1-u) + N1 u) / sqrt(S) dz = dz [N0 2(r0 + 2 r1) + N1 2(r1 + 2 r0)] / (3 (r0 + r1)^2)
        int (N0 (1-u) + N1 u) / S^(3/2) dz = dz [N0 / r0 + N1 / r1] 2 / (r0 + r1)^2
    (both reduce to the trapezoid rule with 1/r, 1/r^3 when S is constant)."""
    q = (r0 + r1) ** 2
    return (
        dz * 2.0 * (r0 + 2.0 * r1) / (3.0 * q),
        dz * 2.0 * (r1 + 2.0 * r0) / (3.0 * q),
        dz * 2.0 / (r0 * q),
        dz * 2.0 / (r1 * q),
    )


def _layer(sh, dz, c0, v0, c1, v1, jac: bool):
    """Integrals over one layer per ray (arrays broadcast). Returns T, X, J, ok."""
    s0, S0, a0, b0 = _point_terms(sh, c0, v0)
    s1, S1, a1, b1 = _point_terms(sh, c1, v1)
    ok = (S0 > 0) & (S1 > 0)
    r0, r1 = np.sqrt(np.where(ok, S0, 1.0)), np.sqrt(np.where(ok, S1, 1.0))
    p0, p1, q0, q1 = (np.where(ok, w, 0.0) for w in layer_weights(r0, r1, dz))
    T = a0 * p0 + a1 * p1
    X = b0 * p0[..., None] + b1 * p1[..., None]
    J = None
    if jac:
        eye = np.eye(2)

        def a1_term(c, v):
            return eye - v[..., :, None] * v[..., None, :] / (c**2)[..., None, None]

        def b3_term(s, c, v):
            left = c[..., None] * sh + s[..., None] * v
            right = (s / c)[..., None] * v + sh
            return left[..., :, None] * right[..., None, :] / c[..., None, None]

        J = (
            a1_term(c0, v0) * p0[..., None, None]
            + a1_term(c1, v1) * p1[..., None, None]
            + b3_term(s0, c0, v0) * q0[..., None, None]
            + b3_term(s1, c1, v1) * q1[..., None, None]
        )
    return T, X, J, ok


def integrate(
    grid: RayGrid,
    sh: FloatArray,
    z_lo: FloatArray,
    z_hi: FloatArray,
    jac: bool = True,
    direction: FloatArray | None = None,
):
    """Ray integrals over [z_lo, z_hi] for each slowness sh (N, 2).

    Returns T (N,), X (N, 2), D and validity, where D is the Jacobian dX/ds_h (N, 2, 2) if
    `jac`, else the directional derivative e^T J e (N,) along unit `direction` e (N, 2) if
    given, else None. Rays are processed in chunks sorted by height so each chunk only
    integrates up to its own highest point.
    """
    sh = np.atleast_2d(np.asarray(sh, dtype=float))
    z_lo = np.broadcast_to(np.asarray(z_lo, dtype=float), (len(sh),))
    z_hi = np.broadcast_to(np.asarray(z_hi, dtype=float), (len(sh),))
    e = None if direction is None else np.atleast_2d(np.asarray(direction, dtype=float))
    n = len(sh)
    T = np.zeros(n)
    X = np.zeros((n, 2))
    D = np.zeros((n, 2, 2)) if jac else (np.zeros(n) if e is not None else None)
    ok = np.ones(n, dtype=bool)
    order = np.argsort(z_hi, kind="stable")
    for lo in range(0, n, _CHUNK):
        idx = order[lo : lo + _CHUNK]
        t, x, d, k = _integrate_chunk(grid, sh[idx], z_lo[idx], z_hi[idx], jac, None if e is None else e[idx])
        T[idx], X[idx], ok[idx] = t, x, k
        if D is not None:
            D[idx] = d
    return T, X, D, ok


def _integrate_chunk(grid: RayGrid, sh, z_lo, z_hi, jac, e):
    """Trapezoid-type sums with exact 1/sqrt(S) layer weights; each node is evaluated once."""
    z = grid.z
    z_lo = np.clip(z_lo, z[0], z[-1])
    z_hi = np.clip(z_hi, z[0], z[-1])
    i_lo = grid.first_node_at_or_above(z_lo)
    i_hi = grid.last_node_at_or_below(z_hi)
    kmax = int(min(max(int(np.max(i_hi)), 0) + 1, len(z) - 1))
    dz = np.diff(z[: kmax + 1])[None, :]  # layer thicknesses
    c_n, v_n = grid.c[: kmax + 1], grid.v[: kmax + 1]
    sabs, S, a, b = _point_terms(sh[:, None, :], c_n[None, :], v_n[None, :, :])
    pos = S > 0
    r = np.sqrt(np.where(pos, S, 1.0))
    k = np.arange(kmax)[None, :]
    inside = (k >= i_lo[:, None]) & (k + 1 <= i_hi[:, None])  # full layers
    layer_ok = pos[:, :-1] & pos[:, 1:]
    good = ~np.any(inside & ~layer_ok, axis=1)
    use = inside & layer_ok
    p0, p1, q0, q1 = (np.where(use, w, 0.0) for w in layer_weights(r[:, :-1], r[:, 1:], dz))
    om1 = np.zeros_like(r)  # node weights accumulated from the layers below and above
    om1[:, :-1] += p0
    om1[:, 1:] += p1
    T = np.sum(om1 * a, axis=1)
    X = np.einsum("nk,nki->ni", om1, b)
    D = None
    if jac or e is not None:
        om3 = np.zeros_like(r)
        om3[:, :-1] += q0
        om3[:, 1:] += q1
        left = c_n[None, :, None] * sh[:, None, :] + sabs[..., None] * v_n[None]
        right = (sabs / c_n[None, :])[..., None] * v_n[None] + sh[:, None, :]
        if jac:
            a1 = np.eye(2)[None] - v_n[:, :, None] * v_n[:, None, :] / (c_n**2)[:, None, None]
            D = np.einsum("nk,kij->nij", om1, a1) + np.einsum(
                "nk,nki,nkj->nij", om3 / c_n[None, :], left, right
            )
        else:
            ve = e @ v_n.T  # (n, k) wind along e
            D = np.sum(om1 * (1.0 - ve**2 / c_n[None, :] ** 2), axis=1) + np.sum(
                om3 / c_n[None, :] * np.einsum("nki,ni->nk", left, e) * np.einsum("nki,ni->nk", right, e),
                axis=1,
            )

    # Partial layers at the ends (or one layer when both ends fall inside the same layer).
    same = i_lo > i_hi
    z_n_lo = np.where(same, z_hi, z[np.clip(i_lo, 0, len(z) - 1)])
    z_n_hi = np.where(same, z_lo, z[np.clip(i_hi, 0, len(z) - 1)])
    for za, zb, used in (
        (z_lo, z_n_lo, (z_n_lo - z_lo > 1e-9) | same),
        (z_n_hi, z_hi, (z_hi - z_n_hi > 1e-9) & ~same),
    ):
        if not np.any(used):
            continue
        ca, va, _ = grid.at(za)
        cb, vb, _ = grid.at(zb)
        t, x, j, kk = _layer(sh, np.abs(zb - za), ca, va, cb, vb, jac or e is not None)
        T += np.where(used, t, 0.0)
        X += np.where(used[:, None], x, 0.0)
        if jac:
            D += np.where(used[:, None, None], j, 0.0)
        elif e is not None:
            D += np.where(used, np.einsum("ni,nij,nj->n", e, j, e), 0.0)
        good &= ~used | kk
    good &= z_hi > z_lo
    return T, X, D, good


def sigma_limit(grid: RayGrid, direction: FloatArray, z_lo: FloatArray, z_hi: FloatArray) -> FloatArray:
    """Largest |s_h| along unit `direction` (N, 2) keeping s_z^2 > 0 on [z_lo, z_hi] (limiting ray)."""
    d = np.atleast_2d(direction)
    out = np.empty(len(d))
    for lo in range(0, len(d), _CHUNK):
        sl = slice(lo, min(lo + _CHUNK, len(d)))
        w = d[sl] @ grid.v.T  # (n, nodes) wind along direction
        lim = 1.0 / (grid.c[None, :] + w)
        inside = (grid.z[None, :] >= z_lo[sl, None] - 1e-9) & (grid.z[None, :] <= z_hi[sl, None] + 1e-9)
        m = np.min(np.where(inside, lim, np.inf), axis=1)
        for zq in (z_lo[sl], z_hi[sl]):
            c, v, _ = grid.at(zq)
            m = np.minimum(m, 1.0 / (c + np.sum(d[sl] * v, axis=1)))
        out[sl] = m
    return out


@dataclass(frozen=True)
class Eigenrays:
    sh: FloatArray  # (N, 2)
    T: FloatArray  # (N,)
    X: FloatArray  # (N, 2)
    J: FloatArray  # (N, 2, 2)
    valid: np.ndarray  # (N,)


def _path_eval(grid, sh, zs, zr, reflected, jac=True, direction=None):
    """Integrals for direct ([min, max] of zs, zr) or reflected ([0, zs] + [0, zr]) paths."""
    if not reflected:
        return integrate(grid, sh, np.minimum(zs, zr), np.maximum(zs, zr), jac, direction)
    t1, x1, j1, k1 = integrate(grid, sh, np.zeros_like(zs), zs, jac, direction)
    t2, x2, j2, k2 = integrate(grid, sh, np.zeros_like(zs), np.full_like(zs, zr), jac, direction)
    if zr <= 0:
        return t1, x1, j1, k1
    return t1 + t2, x1 + x2, (None if j1 is None else j1 + j2), k1 & k2


@dataclass(frozen=True)
class Fan:
    """Windless rays for initial guesses: horizontal range r[k, node] reached at each grid node by
    the ray with horizontal slowness sigma[k] (increasing), for one receiver height and path type."""

    sigma: FloatArray  # (K,)
    r: FloatArray  # (K, n_nodes); NaN where the ray does not reach that height


def build_fan(grid: RayGrid, z_receiver: float, reflected: bool, n_rays: int = 3000) -> Fan:
    """One pass: layer contributions for every ray, then cumulative sums over height."""
    zr = float(np.clip(z_receiver, 0.0, grid.z[-1]))
    v0 = np.zeros_like(grid.v)
    c_r = float(np.interp(zr, grid.z, grid.c))
    # Receiver elevations from grazing to vertical, denser near grazing (where r grows fastest).
    elev = np.radians(np.geomspace(0.01, 90.0, n_rays))[::-1]
    sigma = np.cos(elev) / c_r
    sh = np.column_stack([sigma, np.zeros_like(sigma)])
    dz = np.diff(grid.z)
    _, X, _, ok = _layer(
        sh[:, None, :], dz[None, :], grid.c[None, :-1], v0[None, :-1], grid.c[None, 1:], v0[None, 1:], False
    )
    x = X[..., 0]
    i_r = int(grid.last_node_at_or_below(zr))
    # Partial layer between node i_r and the receiver height.
    zp = grid.z[i_r]
    if zr - zp > 1e-9:
        c_a = np.full(n_rays, float(np.interp(zp, grid.z, grid.c)))
        _, xp, _, okp = _layer(
            sh, zr - zp, c_a, np.zeros((n_rays, 2)), np.full(n_rays, c_r), np.zeros((n_rays, 2)), False
        )
        xp = np.where(okp, xp[:, 0], np.nan)
    else:
        xp = np.zeros(n_rays)
    cum = np.concatenate(
        [np.zeros((n_rays, 1)), np.cumsum(np.where(ok, x, np.nan), axis=1)], axis=1
    )  # from 0
    at_r = cum[:, i_r] + xp  # integral from 0 to the receiver height
    if reflected:
        r = cum + at_r[:, None]  # down to the ground and back up (NaN after any failed layer)
    else:
        # Sources above the receiver only (the fan is just an initial guess; below the receiver,
        # within ~1.5 m of the ground, the straight-line guess is used).
        r = cum - at_r[:, None]
        r[:, : i_r + 1] = np.nan
    return Fan(sigma, r)


def _fan_guess(fan: Fan, grid: RayGrid, zs: FloatArray, r_target: FloatArray) -> FloatArray:
    """Initial |s_h| for each target from the windless fan (NaN if outside the fan)."""
    k = np.clip(np.searchsorted(grid.z, zs) - 1, 0, len(grid.z) - 2)
    w = np.clip((zs - grid.z[k]) / (grid.z[k + 1] - grid.z[k]), 0.0, 1.0)
    out = np.full(len(zs), np.nan)
    for i in range(len(zs)):
        col = (1 - w[i]) * fan.r[:, k[i]] + w[i] * fan.r[:, k[i] + 1]
        good = np.isfinite(col)
        if good.sum() < 2:
            continue
        rc, sg = col[good], fan.sigma[good]
        mono = np.concatenate([[True], np.diff(rc) > 0])  # keep the monotone (direct-ray) prefix
        stop = np.argmin(mono) if not mono.all() else len(mono)
        rc, sg = rc[:stop], sg[:stop]
        if len(rc) >= 2 and rc[0] <= r_target[i] <= rc[-1]:
            out[i] = np.interp(r_target[i], rc, sg)
    return out


def solve_eigenrays(
    grid: RayGrid,
    sources: FloatArray,
    receiver: FloatArray,
    reflected: bool,
    tol: float,
    max_iter: int,
    fan: Fan | None = None,
    initial_sh: FloatArray | None = None,
) -> Eigenrays:
    """Horizontal slowness of the ray from each source to the receiver (direct or ground-reflected).

    `initial_sh` (e.g. the solution for a nearby receiver) starts a 2D Newton iteration directly;
    rays that do not converge from it fall back to the full bracketed solve.
    """
    if initial_sh is not None:
        src_all = np.atleast_2d(np.asarray(sources, dtype=float))
        init = np.asarray(initial_sh, dtype=float)
        n_all = len(src_all)
        sh, T = np.full((n_all, 2), np.nan), np.full(n_all, np.nan)
        X, J, valid = np.zeros((n_all, 2)), np.zeros((n_all, 2, 2)), np.zeros(n_all, dtype=bool)
        warm = np.flatnonzero(np.all(np.isfinite(init), axis=1))
        if len(warm):
            fast = _newton_2d(grid, src_all[warm], receiver, reflected, tol, init[warm], 8)
            sh[warm], T[warm], X[warm], J[warm], valid[warm] = fast.sh, fast.T, fast.X, fast.J, fast.valid
        retry = np.flatnonzero(~valid)
        if len(retry):
            slow = solve_eigenrays(grid, src_all[retry], receiver, reflected, tol, max_iter, fan)
            sh[retry], T[retry], X[retry], J[retry], valid[retry] = (
                slow.sh,
                slow.T,
                slow.X,
                slow.J,
                slow.valid,
            )
        return Eigenrays(sh, T, X, J, valid)
    src = np.atleast_2d(np.asarray(sources, dtype=float))
    n = len(src)
    rcv = np.asarray(receiver, dtype=float)
    zs = np.clip(src[:, 2], 0.0, grid.z[-1])
    zr = float(np.clip(rcv[2], 0.0, grid.z[-1]))
    delta = rcv[None, :2] - src[:, :2]  # required displacement source -> receiver
    r = np.linalg.norm(delta, axis=1)
    e = np.where(r[:, None] > 1e-9, delta / np.maximum(r, 1e-300)[:, None], np.array([1.0, 0.0]))
    z_lo = np.zeros(n) if reflected else np.minimum(zs, zr)
    z_hi = np.maximum(zs, zr)
    hi = sigma_limit(grid, e, z_lo, z_hi) * (1 - _EPS_SIGMA)
    lo = -sigma_limit(grid, -e, z_lo, z_hi) * (1 - _EPS_SIGMA)

    # Stage 1: bracketed Newton on sigma along e, f(sigma) = X(sigma e) . e - r (increasing).
    def f_and_df(sig, idx):
        _, X, dfd, ok = _path_eval(grid, sig[:, None] * e[idx], zs[idx], zr, reflected, False, e[idx])
        return np.sum(X * e[idx], axis=1) - r[idx], dfd, ok

    c_r, _, _ = grid.at(np.array([zr]))
    height = np.maximum(z_hi - z_lo, 1e-6)
    straight = r / np.hypot(r, height if not reflected else zs + zr) / c_r[0]
    guess = _fan_guess(fan, grid, zs, r) if fan is not None else np.full(n, np.nan)
    sig = np.clip(np.where(np.isfinite(guess), guess, straight), lo, hi)
    active = np.ones(n, dtype=bool)
    shadow = np.zeros(n, dtype=bool)
    hi_checked = np.zeros(n, dtype=bool)
    for it in range(max_iter):
        if not np.any(active):
            break
        if it == 6:  # slow convergence suggests a target beyond the limiting ray: test it once
            idx = np.flatnonzero(active & ~hi_checked)
            if len(idx):
                f_hi, _, ok_hi = f_and_df(hi[idx], idx)
                beyond = ~ok_hi | (f_hi < 0)
                shadow[idx[beyond]] = True
                active[idx[beyond]] = False
                hi_checked[idx] = True
                if not np.any(active):
                    break
        idx = np.flatnonzero(active)
        fv, dfv, okv = f_and_df(sig[idx], idx)
        fv = np.where(okv, fv, np.inf)  # an invalid ray is past the limit: treat as too far
        neg = fv < 0
        lo[idx] = np.where(neg, sig[idx], lo[idx])
        hi[idx] = np.where(~neg, sig[idx], hi[idx])
        done = np.abs(fv) < tol
        newton = sig[idx] - np.where(np.isfinite(fv) & (dfv > 0), fv / np.where(dfv > 0, dfv, 1.0), np.nan)
        inside = np.isfinite(newton) & (newton > lo[idx]) & (newton < hi[idx])
        sig[idx] = np.where(done, sig[idx], np.where(inside, newton, 0.5 * (lo[idx] + hi[idx])))
        active[idx[done]] = False
    converged_1d = ~active & ~shadow

    # Stage 2: 2D Newton for the crosswind drift (no-op without wind).
    sh = sig[:, None] * e
    T, X, J, ok = _path_eval(grid, sh, zs, zr, reflected)
    if np.any(np.abs(grid.v) > 0):
        for _ in range(max_iter):
            g = X - delta
            need = converged_1d & ok & (np.linalg.norm(g, axis=1) >= tol)
            if not np.any(need):
                break
            idx = np.flatnonzero(need)
            step = -np.linalg.solve(J[idx], g[idx][..., None])[..., 0]
            Tt, Xt, Jt, okt = _path_eval(grid, sh[idx] + step, zs[idx], zr, reflected)
            g_norm = np.linalg.norm(g[idx], axis=1)
            for _ in range(20):  # backtrack (only the rays whose trial is invalid or not closer)
                worse = ~(okt & (np.linalg.norm(Xt - delta[idx], axis=1) < g_norm))
                if not np.any(worse):
                    break
                step[worse] *= 0.5
                w = np.flatnonzero(worse)
                Tt[w], Xt[w], Jt[w], okt[w] = _path_eval(
                    grid, sh[idx][w] + step[w], zs[idx][w], zr, reflected
                )
            sh[idx], T[idx], X[idx], J[idx], ok[idx] = sh[idx] + step, Tt, Xt, Jt, okt
    valid = converged_1d & ok & (np.linalg.norm(X - delta, axis=1) < 10 * tol)
    return Eigenrays(sh, np.where(valid, T, np.nan), X, J, valid)


def _newton_2d(grid, sources, receiver, reflected, tol, sh0, max_iter) -> Eigenrays:
    """2D Newton on s_h from sh0 with backtracking; rays that fail are returned invalid."""
    src = np.atleast_2d(np.asarray(sources, dtype=float))
    rcv = np.asarray(receiver, dtype=float)
    zs = np.clip(src[:, 2], 0.0, grid.z[-1])
    zr = float(np.clip(rcv[2], 0.0, grid.z[-1]))
    delta = rcv[None, :2] - src[:, :2]
    sh = sh0.copy()
    T, X, J, ok = _path_eval(grid, sh, zs, zr, reflected)
    for _ in range(max_iter):
        g = X - delta
        need = ok & (np.linalg.norm(g, axis=1) >= tol)
        if not np.any(need):
            break
        idx = np.flatnonzero(need)
        step = -np.linalg.solve(J[idx], g[idx][..., None])[..., 0]
        Tt, Xt, Jt, okt = _path_eval(grid, sh[idx] + step, zs[idx], zr, reflected)
        g_norm = np.linalg.norm(g[idx], axis=1)
        for _ in range(10):
            worse = ~(okt & (np.linalg.norm(Xt - delta[idx], axis=1) < g_norm))
            if not np.any(worse):
                break
            step[worse] *= 0.5
            w = np.flatnonzero(worse)
            Tt[w], Xt[w], Jt[w], okt[w] = _path_eval(grid, sh[idx][w] + step[w], zs[idx][w], zr, reflected)
        sh[idx], T[idx], X[idx], J[idx], ok[idx] = sh[idx] + step, Tt, Xt, Jt, okt
    valid = ok & (np.linalg.norm(X - delta, axis=1) < 10 * tol)
    return Eigenrays(sh, np.where(valid, T, np.nan), X, J, valid)


def ray_geometry(grid: RayGrid, rays: Eigenrays, sources: FloatArray, receiver: FloatArray, reflected: bool):
    """Arrival wave normal at the receiver, source slowness vector and spreading amplitude."""
    src = np.atleast_2d(np.asarray(sources, dtype=float))
    zs = np.clip(src[:, 2], 0.0, grid.z[-1])
    zr = float(np.clip(receiver[2], 0.0, grid.z[-1]))
    sh = rays.sh
    c_s, v_s, rho_s = grid.at(zs)
    c_r, v_r, rho_r = grid.at(np.full_like(zs, zr))
    s_s, S_s, _, _ = _point_terms(sh, c_s, v_s)
    s_r, S_r, _, _ = _point_terms(sh, c_r, v_r)
    up_s = (not reflected) & (zs < zr)  # the ray leaves the source upward only if it is below the receiver
    sz_s = np.where(up_s, 1.0, -1.0) * np.sqrt(np.clip(S_s, 0.0, None))
    up_r = reflected | (zs < zr)  # ... and arrives travelling upward if reflected or from below
    sz_r = np.where(up_r, 1.0, -1.0) * np.sqrt(np.clip(S_r, 0.0, None))
    n_r = np.column_stack([sh, sz_r]) / s_r[:, None]
    slow_s = np.column_stack([sh, sz_s])
    det_n = (1.0 + np.sum(v_s * sh, axis=1) / (c_s * s_s)) / s_s**2
    nz_s = np.abs(sz_s) / s_s
    g = c_r[:, None] * n_r + np.column_stack([v_r, np.zeros(len(sh))])
    gz = np.abs(g[:, 2]) / np.linalg.norm(g, axis=1)
    det_j = np.abs(np.linalg.det(rays.J))
    with np.errstate(divide="ignore", invalid="ignore"):
        amp2 = (rho_r * c_r) / (rho_s * c_s) * np.abs(det_n) / (nz_s * det_j * gz)
        amp = np.sqrt(amp2)
    return n_r, slow_s, amp


def locate(grid: RayGrid, receiver: FloatArray, sh: FloatArray, travel_time: FloatArray):
    """Positions reached by tracing rays with slowness sh up from the receiver for travel_time."""
    sh = np.atleast_2d(np.asarray(sh, dtype=float))
    tt = np.asarray(travel_time, dtype=float)
    n = len(sh)
    zr = float(np.clip(receiver[2], 0.0, grid.z[-1]))
    pos = np.full((n, 3), np.nan)
    valid = np.zeros(n, dtype=bool)
    z = grid.z
    i0 = int(grid.first_node_at_or_above(zr))
    dzs = np.diff(z[i0:])
    for lo in range(0, n, _CHUNK):
        sl = slice(lo, min(lo + _CHUNK, n))
        s = sh[sl]
        # first (partial) layer from the receiver to the next node, then full layers upward
        c_a, v_a, _ = grid.at(np.full(len(s), zr))
        if z[i0] > zr + 1e-9:
            t0, x0, _, ok0 = _layer(s, z[i0] - zr, c_a, v_a, grid.c[i0], grid.v[i0], False)
        else:
            t0, x0, ok0 = np.zeros(len(s)), np.zeros((len(s), 2)), np.ones(len(s), bool)
        T, X, _, ok = _layer(
            s[:, None, :],
            dzs[None, :],
            grid.c[None, i0:-1],
            grid.v[None, i0:-1],
            grid.c[None, i0 + 1 :],
            grid.v[None, i0 + 1 :],
            False,
        )
        bad = np.cumsum(~ok, axis=1) > 0  # once a layer fails (turning point), all above fail
        T = np.where(bad, np.inf, T)
        cumT = t0[:, None] + np.concatenate([np.zeros((len(s), 1)), np.cumsum(T, axis=1)], axis=1)
        cumX = x0[:, None, :] + np.concatenate(
            [np.zeros((len(s), 1, 2)), np.cumsum(np.where(bad[..., None], 0, X), axis=1)], axis=1
        )
        heights = z[i0:]
        target = tt[sl]
        k = np.argmax(cumT >= target[:, None], axis=1)  # first node at or past the travel time
        reached = (cumT[np.arange(len(s)), k] >= target) & ok0 & (target > t0) & (k > 0)
        k = np.clip(k, 1, cumT.shape[1] - 1)
        rows = np.arange(len(s))
        ta, tb = cumT[rows, k - 1], cumT[rows, k]
        frac = np.clip((target - ta) / np.where(tb > ta, tb - ta, 1.0), 0.0, 1.0)
        zz = heights[k - 1] + frac * (heights[k] - heights[k - 1])
        xx = cumX[rows, k - 1] + frac[:, None] * (cumX[rows, k] - cumX[rows, k - 1])
        # The displacement integral runs source -> receiver, so the source sits at receiver - X.
        pos[sl] = np.column_stack([receiver[0] - xx[:, 0], receiver[1] - xx[:, 1], zz])
        valid[sl] = reached & np.isfinite(tb)
    return pos, valid


def trace_ray(
    atm,
    start: FloatArray,
    direction: FloatArray,
    t_max: float,
    z_stop: float | None = None,
    max_step: float = 0.05,
):
    """Reference ray from `start` with initial unit wave normal `direction`, by solve_ivp.

    Integrates the Hamiltonian ray equations (wind and sound-speed gradients by central
    differences of the profiles). Stops at the ground, at t_max, or when descending through
    z_stop. Returns (t, positions (k, 3), slowness (k, 3)).
    """
    def deriv(f, z):
        # Central difference with a step that shrinks near the ground, where the power-law
        # wind profile curves sharply.
        h = min(0.25, max(1e-4, 0.01 * (abs(z) + 0.1)))
        return (f(np.array([z + h])) - f(np.array([z - h]))) / (2 * h)

    def rhs(t, y):
        z = y[2]
        s = y[3:]
        c = float(atm.sound_speed(np.array([z]))[0])
        v = atm.wind(np.array([z]))[0]
        sabs = np.linalg.norm(s)
        dc = float(deriv(atm.sound_speed, z)[0])
        dv = deriv(atm.wind, z)[0]
        return np.concatenate([c * s / sabs + v, [0.0, 0.0, -(sabs * dc + s[0] * dv[0] + s[1] * dv[1])]])

    n0 = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    c0 = float(atm.sound_speed(np.array([start[2]]))[0])
    v0 = atm.wind(np.array([start[2]]))[0]
    s0 = n0 / (c0 + v0 @ n0)

    def ground(t, y):
        return y[2]

    events = [ground]
    if z_stop is not None:

        def stop(t, y):
            return y[2] - z_stop

        events.append(stop)
    for ev in events:  # solve_ivp event attributes: stop integrating when z decreases through 0
        setattr(ev, "terminal", True)  # noqa: B010
        setattr(ev, "direction", -1)  # noqa: B010
    sol = solve_ivp(
        rhs,
        (0.0, t_max),
        np.concatenate([start, s0]),
        method="DOP853",
        rtol=1e-10,
        atol=1e-10,
        events=events,
        max_step=max_step * t_max if t_max > 0 else np.inf,
        dense_output=False,
    )
    return sol.t, sol.y[:3].T, sol.y[3:].T
