"""Phase 5 Method D: Bayesian reconstruction with uncertainty and atmospheric self-calibration.

Observations. Per analysis window k the same matched-window TDOAs as Methods A and C give
each mic's delay d_m behind the reference mic; with the reference arrival time t_c and the
reported flash time t0, every mic has an observed travel time  t_km = t_c + d_m - t0.

Likelihood. t_k ~ N(T(x_k; atm) + dt0_g(k), Sigma_k), with
    Sigma_k = a_k I + b_k 1 1^T,
    a_k = sigma_meas,k^2 + sigma_clock^2 + (sigma_position / c)^2      (independent per mic)
    b_k = spread_k^2  (+ sigma_t0^2 when dt0 is not estimated)          (common to the window)
sigma_meas,k comes from the window's own pair-TDOA misfit with a floor (`d_sigma_timing_s`);
spread_k is the RMS time spread of the window's sound at the reference mic (which part of an
extended source the point stands for). A position error moves a mic along the ray by
~sigma_position, i.e. sigma_position / c of time. dt0_g is the flash-time error of recording g.

Forward models.
  * Effective medium (self-calibration): straight rays through a uniformly moving medium whose
    parameters depend on the source height h,  c(h) = c0 + c1 h,  w(h) = w0 + w1 h  (w
    horizontal). These are *path-averaged* values: wind grows with height, so sound from a
    higher source crosses faster air on average. The travel time solves |d - w T| = c T,
    d = mic - source, in closed form, with analytic derivatives (implicit differentiation).
    With the parameters fitted to the true stratified atmosphere it nearly reproduces the
    oracle (development bolt: 1.4 m in still air, 7.7 m in 6 m/s wind; docs/log.md).
  * Fixed atmosphere: any assumed Atmosphere (straight or traced rays) via `propagate`, as in
    Method C. The windows are then independent; sigma_t0 enters each window's covariance.
  Parameters theta = (c0, c1, w0x, w0y, w1x, w1y, dt0_1 .. dt0_G) have Gaussian priors centered
  on the assumed atmosphere (surface sound speed and the path-averaged gradient and wind) and
  on the reported flash times; poorly constrained parameters stay near their priors.

Storm self-calibration (`reconstruct_bayes_storm`): several recordings share the atmosphere
(each keeps its own dt0). One bolt is seen over a narrow range of azimuths, so only the wind
along the line of sight changes anything measurable; a cross-wind shifts every apparent source
sideways in proportion to its travel time, which looks exactly like a slightly different
channel. Bolts at other azimuths see that wind component along their line of sight.

Inference. MAP over all window positions and theta by variable projection: given theta the
windows separate into 3-parameter problems (solved together, vectorized; poorly fitting
windows are retried from their original start), and an outer trust-region solver (scipy, exact
solves) works on theta alone with the exactly projected Jacobian. Three stages: positions at
the prior; joint with a soft-L1 loss (outlier tolerant); then, after a chi-square gate
(p < d_outlier_p) drops inconsistent windows, a plain Gaussian refit of the inliers so the
Laplace approximation describes the model actually assumed. Laplace covariances: theta from
the Schur complement S^-1 (with the prior); each point from its block of the inverse joint
Hessian, H_xx^-1 + H_xx^-1 H_xt S^-1 H_tx H_xx^-1, so the atmosphere's uncertainty propagates
into every point. MCMC (emcee) per window, given the MAP theta, plus the same propagation term,
is the slow reference (d_inference = "mcmc").

Calibration (tests/test_bayes.py, docs/log.md): on synthetic observations drawn from this
likelihood the 1/2/3-sigma coverage is nominal (19.9 / 73.9 / 97.1%) with a fixed atmosphere
on any array and with self-calibration on arrays with vertical aperture. With self-calibration
on a planar array the 3-sigma tail is light (~91%): a planar array measures only horizontal
slowness, so a source's elevation follows from cos(el) = c |s_h|, which is extremely sensitive
to c near the horizon; the posterior there is not Gaussian.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats
from threadpoolctl import threadpool_limits

from thunder.config import ReconstructionConfig
from thunder.recon.multilat import relative_delays
from thunder.recon.tdoa import Setup, WindowFit, assemble, method_a_fits, place_points, prepare
from thunder.types import ArrivalPath, Atmosphere, FloatArray, Reconstruction, Recording

ATM_NAMES = ("c0", "c1", "w0x", "w0y", "w1x", "w1y")
N_ATM = len(ATM_NAMES)
PRIOR_HEIGHTS_M = np.linspace(0.0, 8000.0, 81)  # for the path-averaged prior from an atmosphere

# model(x (K, 3), medium (6,), mic offsets (M, 3)) -> T (K, M), dT/dx (K, M, 3), dT/dmedium (K, M, 6)
Model = Callable[[FloatArray, FloatArray, FloatArray], tuple[FloatArray, FloatArray, FloatArray]]


# --- forward models -------------------------------------------------------------------


def effective_times(
    x: FloatArray, mics: FloatArray, atm: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Travel times (K, M), dT/dx (K, M, 3) and dT/datm (K, M, 6) in the effective medium.

    Solves |d - w T| = c T for T > 0 (d = mic - source): sound moves at c relative to air that
    itself moves at w.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    c0, c1, w0x, w0y, w1x, w1y = np.asarray(atm, dtype=float)[:N_ATM]
    h = np.maximum(x[:, 2], 0.0)[:, None]  # (K, 1)
    above = (x[:, 2] > 0.0)[:, None]
    c = c0 + c1 * h  # (K, 1)
    w = np.zeros((len(x), 1, 3))
    w[..., 0] = w0x + w1x * h
    w[..., 1] = w0y + w1y * h
    d = mics[None, :, :] - x[:, None, :]  # (K, M, 3)
    dw = np.einsum("kmi,kmi->km", d, np.broadcast_to(w, d.shape))
    w2 = np.sum(w**2, axis=-1)  # (K, 1)
    dd = np.sum(d**2, axis=-1)
    den = c**2 - w2  # > 0 for any physical medium (wind slower than sound)
    with np.errstate(invalid="ignore", divide="ignore"):
        T = (-dw + np.sqrt(dw**2 + den * dd)) / den
        g = d - w * T[..., None]  # air-frame path
        f_t = -2.0 * np.einsum("kmi,kmi->km", g, np.broadcast_to(w, g.shape)) - 2.0 * c**2 * T
        dT_dd = -2.0 * g / f_t[..., None]  # dT/dd = -F_d / F_T
        dT_dc = 2.0 * c * T**2 / f_t  # -F_c / F_T with F_c = -2 c T^2
        dT_dw = 2.0 * T[..., None] * g / f_t[..., None]  # -F_w / F_T with F_w = -2 T g
    jx = -dT_dd  # d = m - x
    # Height dependence of c and w (only above ground, where h = z).
    dz = dT_dc * c1 + dT_dw[..., 0] * w1x + dT_dw[..., 1] * w1y
    jx[..., 2] += np.where(above, dz, 0.0)
    jatm = np.stack(
        [dT_dc, dT_dc * h, dT_dw[..., 0], dT_dw[..., 1], dT_dw[..., 0] * h, dT_dw[..., 1] * h], axis=-1
    )
    return T, jx, jatm


class AtmosphereModel:
    """Fixed assumed atmosphere (straight or traced rays): T and dT/dx = -source slowness."""

    def __init__(self, atmosphere: Atmosphere, mics: FloatArray):
        self.atmosphere = atmosphere
        self.mics = mics
        self.guesses: list[ArrivalPath | None] = [None] * len(mics)

    def __call__(
        self, x: FloatArray, atm: FloatArray, mic_offsets: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        # Mic offsets are not estimated with a fixed atmosphere (they are noise terms there).
        # Warm starts only fit the same set of sources (retries use a subset).
        guesses = [g if g is not None and len(g.travel_time) == len(x) else None for g in self.guesses]
        paths = [self.atmosphere.propagate(x, m, guess=g) for m, g in zip(self.mics, guesses, strict=True)]
        self.guesses = list(paths)
        T = np.stack([p.travel_time for p in paths], axis=1)
        jx = -np.stack([p.source_slowness for p in paths], axis=1)
        return T, jx, np.zeros((*T.shape, N_ATM))


def prior_from_atmosphere(atmosphere: Atmosphere, cfg: ReconstructionConfig) -> tuple[FloatArray, FloatArray]:
    """Prior mean and standard deviation of the 6 medium parameters from the assumed atmosphere.

    Mean: the straight-line fit of the *path-averaged* sound speed and wind (mean over [0, h])
    versus source height h up to 8 km. Std: config.
    """
    z = PRIOR_HEIGHTS_M
    c = atmosphere.sound_speed(z)
    wind = atmosphere.wind(z)[:, :2]

    def path_mean(v: FloatArray) -> FloatArray:
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(z))])
        out = np.empty_like(v)
        out[0] = v[0]
        out[1:] = cum[1:] / z[1:]
        return out

    def line(v: FloatArray) -> tuple[float, float]:
        slope, intercept = np.polyfit(z, path_mean(v), 1)
        return float(intercept), float(slope)

    c0, c1 = line(c)
    w0x, w1x = line(wind[:, 0])
    w0y, w1y = line(wind[:, 1])
    mean = np.array([c0, c1, w0x, w0y, w1x, w1y])
    std = np.array(
        [
            cfg.d_prior_c_mps,
            cfg.d_prior_c_gradient,
            cfg.d_prior_wind_mps,
            cfg.d_prior_wind_mps,
            cfg.d_prior_wind_gradient,
            cfg.d_prior_wind_gradient,
        ]
    )
    return mean, std


# --- inference ------------------------------------------------------------------------


@dataclass
class NoiseModel:
    """Per-window covariance a_k I + b_k 1 1^T (see module docstring)."""

    a: FloatArray  # (K,)
    b: FloatArray  # (K,)

    def whiten(self, v: FloatArray) -> FloatArray:
        """Sigma^-1/2 v for v of shape (K, M, ...): a^-1/2 (v - mean) + (a + M b)^-1/2 mean."""
        m = v.shape[1]
        shape = (-1,) + (1,) * (v.ndim - 1)
        mean = v.mean(axis=1, keepdims=True)
        alpha = (1.0 / np.sqrt(self.a)).reshape(shape)
        beta = (1.0 / np.sqrt(self.a + m * self.b)).reshape(shape)
        return alpha * (v - mean) + beta * mean

    def covariance(self, k: int, m: int) -> FloatArray:
        return self.a[k] * np.eye(m) + self.b[k] * np.ones((m, m))

    def subset(self, mask: np.ndarray) -> NoiseModel:
        return NoiseModel(self.a[mask], self.b[mask])


@dataclass
class Problem:
    """Stacked windows of one or more recordings and the parameter layout.

    theta = [medium (6) | dt0 per recording (G) | if array_cal: clock offset per mic (M) |
    position offset per mic (3M)]. Array calibration models per-mic errors that are the same in
    every window: left as independent noise they look like signal once hundreds of windows are
    pooled, and the shared medium parameters absorb them (E7).
    """

    t_obs: FloatArray  # (K, M) observed travel times
    noise: NoiseModel
    group: np.ndarray  # (K,) recording index of each window (its dt0)
    model: Model
    prior_mean: FloatArray
    prior_std: FloatArray
    free: np.ndarray  # bool, same length as theta
    array_cal: bool = False

    @property
    def n_mics(self) -> int:
        return int(self.t_obs.shape[1])

    @property
    def n_groups(self) -> int:
        return len(self.prior_mean) - N_ATM - (4 * self.n_mics if self.array_cal else 0)

    def subset(self, mask: np.ndarray) -> Problem:
        return dataclasses.replace(
            self, t_obs=self.t_obs[mask], noise=self.noise.subset(mask), group=self.group[mask]
        )

    def split(self, theta: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """(medium, dt0 per recording, clock offset per mic, position offset per mic (M, 3))."""
        g, m = self.n_groups, self.n_mics
        atm, dt0 = theta[:N_ATM], theta[N_ATM : N_ATM + g]
        if not self.array_cal:
            return atm, dt0, np.zeros(m), np.zeros((m, 3))
        k = N_ATM + g
        return atm, dt0, theta[k : k + m], theta[k + m : k + 4 * m].reshape(m, 3)

    def evaluate(self, x: FloatArray, theta: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Residuals (K, M), dT/dx (K, M, 3) and d(prediction)/dtheta (K, M, len(theta))."""
        atm, dt0, clock, dm = self.split(theta)
        T, jx, jatm = self.model(x, atm, dm)
        k, m = T.shape
        r = self.t_obs - T - dt0[self.group][:, None] - clock[None, :]
        onehot = np.zeros((k, m, self.n_groups))
        onehot[np.arange(k), :, self.group] = 1.0
        blocks = [jatm, onehot]
        if self.array_cal:
            blocks.append(np.broadcast_to(np.eye(m), (k, m, m)))  # clock offset of mic j
            jpos = np.zeros((k, m, 3 * m))
            for j in range(m):  # moving mic j by dm changes its time by dT/dm = -dT/dx
                jpos[:, j, 3 * j : 3 * j + 3] = -jx[:, j, :]
            blocks.append(jpos)
        return r, jx, np.concatenate(blocks, axis=-1)


@dataclass
class JointFit:
    x: FloatArray  # (K, 3)
    theta: FloatArray  # (6 + G,)
    cov_x: FloatArray  # (K, 3, 3)
    cov_theta: FloatArray  # (6 + G, 6 + G) (zero rows/columns for fixed parameters)
    q: FloatArray  # (K,) Mahalanobis norm^2 of each window's residual
    converged: bool
    evaluations: int


def conditional_positions(
    pb: Problem, x0: FloatArray, theta: FloatArray, max_iter: int = 30
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Every window's position given theta: independent 3-parameter Gauss-Newton problems,
    solved together (vectorized), with per-window Levenberg damping and step acceptance.
    Returns positions (K, 3), q = Mahalanobis norm^2 (K,) and H_xx (K, 3, 3) at the solution."""
    x = np.array(x0, dtype=float)
    k = len(x)
    lam = np.full(k, 1e-6)

    def evaluate(xx):
        with np.errstate(invalid="ignore", divide="ignore"):
            r, jx, _ = pb.evaluate(xx, theta)
            rt = pb.noise.whiten(r)
            q = np.sum(rt**2, axis=1)
            jxt = pb.noise.whiten(jx)
        return np.where(np.isfinite(q), q, np.inf), np.where(np.isfinite(rt), rt, 0.0), np.nan_to_num(jxt)

    q, rt, jxt = evaluate(x)
    active = np.isfinite(q)
    for _ in range(max_iter):
        h = np.einsum("kmi,kmj->kij", jxt, jxt)
        g = np.einsum("kmi,km->ki", jxt, rt)
        damp = lam[:, None, None] * np.einsum("kii->ki", h)[:, :, None] * np.eye(3)[None]
        dx = np.linalg.solve(h + damp + 1e-30 * np.eye(3)[None], g[..., None])[..., 0]
        dx = np.where(active[:, None] & np.isfinite(dx), dx, 0.0)
        q_t, rt_t, jxt_t = evaluate(x + dx)
        better = active & (q_t <= q)
        x = np.where(better[:, None], x + dx, x)
        q = np.where(better, q_t, q)
        rt = np.where(better[:, None], rt_t, rt)
        jxt = np.where(better[:, None, None], jxt_t, jxt)
        lam = np.where(better, np.maximum(lam * 0.1, 1e-12), lam * 10.0)
        active &= ~((better & (np.max(np.abs(dx), axis=1) < 1e-4)) | (lam > 1e8))
        if not active.any():
            break
    return x, q, np.einsum("kmi,kmj->kij", jxt, jxt)


def best_positions(
    pb: Problem,
    x_warm: FloatArray,
    x_first: FloatArray,
    theta: FloatArray,
    retry_q: float,
    max_iter: int = 30,
) -> FloatArray:
    """Positions given theta from the warm start; windows that fit badly (q > retry_q) are
    re-solved from their original start and keep the better fit. A window pushed to the horizon
    while theta was far off (cos el = c |s_h| > 1 has no solution) would otherwise stay in that
    local minimum after theta improves."""
    x, q, _ = conditional_positions(pb, x_warm, theta, max_iter)
    bad = ~(q <= retry_q)
    if bad.any():
        xb, qb, _ = conditional_positions(pb.subset(bad), x_first[bad], theta, max_iter)
        better = qb < q[bad]
        x[np.flatnonzero(bad)[better]] = xb[better]
    return x


def laplace(pb: Problem, x: FloatArray, theta: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Laplace covariances at (x, theta): points (K, 3, 3), theta, and q (K,).

    Joint Hessian blocks (Gaussian likelihood); theta by the Schur complement S^-1 (with the
    prior), each point by its block of the joint inverse, H_xx^-1 + H_xx^-1 H_xt S^-1 H_tx H_xx^-1.
    """
    idx = np.flatnonzero(pb.free)
    with np.errstate(invalid="ignore", divide="ignore"):
        r, jx, jth = pb.evaluate(x, theta)
        q = np.sum(pb.noise.whiten(r) ** 2, axis=1)
        jxt = np.nan_to_num(pb.noise.whiten(jx))
        jtt = np.nan_to_num(pb.noise.whiten(jth[..., idx]))
    q = np.where(np.isfinite(q), q, np.inf)
    hxx = np.einsum("kmi,kmj->kij", jxt, jxt)
    hxx_inv = np.linalg.pinv(hxx, hermitian=True)
    cov_theta = np.zeros((len(theta), len(theta)))
    cov_x = hxx_inv.copy()
    if len(idx):
        hxt = np.einsum("kmi,kmj->kij", jxt, jtt)
        htt = np.einsum("kmi,kmj->ij", jtt, jtt)
        s = htt + np.diag(1.0 / pb.prior_std[idx] ** 2) - np.einsum("kji,kjl,klm->im", hxt, hxx_inv, hxt)
        s_inv = np.linalg.pinv(s, hermitian=True)
        a_term = np.einsum("kij,kjl->kil", hxx_inv, hxt)  # H_xx^-1 H_xt
        cov_x = cov_x + np.einsum("kil,lm,kjm->kij", a_term, s_inv, a_term)
        cov_theta[np.ix_(idx, idx)] = s_inv
    return cov_x, cov_theta, q


def solve_joint(
    pb: Problem,
    x0: FloatArray,
    cfg: ReconstructionConfig,
    robust: bool = True,
    theta0: FloatArray | None = None,
    x_first: FloatArray | None = None,
) -> JointFit:
    """MAP of all window positions and the free theta (variable projection), with Laplace
    covariances.

    The positions separate given theta, so the problem is solved over theta alone: every
    evaluation re-solves all windows (warm-started), and the outer Jacobian is the whitened
    theta-Jacobian projected orthogonally to each window's position directions (exact at the
    inner optimum; Golub-Pereyra / Kaufman). The outer problem has 6 + G unknowns: scipy's
    trust-region solver with exact solves. robust: soft-L1 loss on the whitened residuals
    (scale 3 sigma), for the first, outlier-tolerant pass.
    """
    from scipy.optimize import least_squares

    idx = np.flatnonzero(pb.free)
    theta_base = np.array(pb.prior_mean if theta0 is None else theta0, dtype=float)
    first = np.array(x0 if x_first is None else x_first, dtype=float)
    retry_q = float(stats.chi2.ppf(1.0 - cfg.d_outlier_p, df=max(pb.t_obs.shape[1] - 3, 1)))
    state = {"x": np.array(x0, dtype=float)}

    def theta_of(u: FloatArray) -> FloatArray:
        th = theta_base.copy()
        th[idx] = u
        return th

    def inner(u: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        th = theta_of(u)
        state["x"] = best_positions(pb, state["x"], first, th, retry_q)
        with np.errstate(invalid="ignore", divide="ignore"):
            r, jx, jth = pb.evaluate(state["x"], th)
            rt = pb.noise.whiten(r)
            jxt = np.nan_to_num(pb.noise.whiten(jx))
            jtt = np.nan_to_num(pb.noise.whiten(jth[..., idx]))
        return np.where(np.isfinite(rt), rt, 1e6), jxt, jtt

    def residuals(u: FloatArray) -> FloatArray:
        rt, _, _ = inner(u)
        return np.concatenate([rt.ravel(), (u - pb.prior_mean[idx]) / pb.prior_std[idx]])

    def jacobian(u: FloatArray) -> FloatArray:
        _, jxt, jtt = inner(u)
        # d r / d theta = -(I - Jx (Jx^T Jx)^-1 Jx^T) J_theta per window (r = observed - model).
        h = np.einsum("kmi,kmj->kij", jxt, jxt)
        proj = jtt - np.einsum("kmi,kij,knj,knp->kmp", jxt, np.linalg.pinv(h, hermitian=True), jxt, jtt)
        return np.vstack([-proj.reshape(-1, len(idx)), np.diag(1.0 / pb.prior_std[idx])])

    if len(idx):
        res = least_squares(
            residuals,
            theta_base[idx],
            jac=jacobian,
            method="trf",
            x_scale=pb.prior_std[idx],
            loss="soft_l1" if robust else "linear",
            f_scale=3.0,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
            max_nfev=cfg.d_max_iterations,
        )
        theta, converged, nfev = theta_of(res.x), bool(res.status > 0), int(res.nfev)
    else:
        theta, converged, nfev = theta_base, True, 0
    x = best_positions(pb, state["x"], first, theta, retry_q, max_iter=60)
    cov_x, cov_theta, q = laplace(pb, x, theta)
    return JointFit(x, theta, cov_x, cov_theta, q, converged, nfev)


def infer(pb: Problem, x0: FloatArray, cfg: ReconstructionConfig) -> tuple[JointFit, np.ndarray]:
    """Full inference: robust joint fit, chi-square gate, plain refit of the inliers.

    Returns the fit (all windows; covariances valid for inliers) and the inlier mask."""
    limit = float(stats.chi2.ppf(1.0 - cfg.d_outlier_p, df=max(pb.t_obs.shape[1] - 3, 1)))
    if not pb.free.any():  # fixed atmosphere: the windows are independent
        fit = solve_joint(pb, x0, cfg)
        return fit, np.isfinite(fit.q) & (fit.q <= limit) & (fit.x[:, 2] > -50.0)
    fixed = dataclasses.replace(pb, free=np.zeros_like(pb.free))
    fit = solve_joint(fixed, x0, cfg)  # positions at the prior
    fit = solve_joint(pb, fit.x, cfg, theta0=fit.theta, x_first=x0)  # joint, robust
    inlier = np.isfinite(fit.q) & (fit.q <= limit) & (fit.x[:, 2] > -50.0)
    if not inlier.any():
        return fit, inlier
    fit_in = solve_joint(
        pb.subset(inlier), fit.x[inlier], cfg, robust=False, theta0=fit.theta, x_first=x0[inlier]
    )
    q = np.full(len(inlier), np.inf)
    q[inlier] = fit_in.q
    x = fit.x.copy()
    x[inlier] = fit_in.x
    cov = np.zeros_like(fit.cov_x)
    cov[inlier] = fit_in.cov_x
    full = JointFit(x, fit_in.theta, cov, fit_in.cov_theta, q, fit_in.converged, fit_in.evaluations)
    return full, inlier & (q <= limit)


def window_noise(
    st_c: float,
    misfit: FloatArray,
    spread: FloatArray,
    n_pairs: int,
    m: int,
    cfg: ReconstructionConfig,
    dt0_free: bool,
    array_cal: bool = False,
) -> NoiseModel:
    """Per-window noise model from each window's pair misfit (with a floor) and time spread.
    With array calibration the clock and position errors are parameters, not noise."""
    dof = max(n_pairs - (m - 1), 1)
    sigma_meas = np.maximum(misfit * math.sqrt(n_pairs / (2.0 * dof)), cfg.d_sigma_timing_s)
    a = sigma_meas**2
    if not array_cal:
        a = a + cfg.d_sigma_clock_s**2 + (cfg.d_sigma_position_m / st_c) ** 2
    b = spread**2 + (0.0 if dt0_free else cfg.d_sigma_t0_s**2)
    return NoiseModel(a, b)


def conditional_cov(pb: Problem, x: FloatArray, theta: FloatArray) -> FloatArray:
    """Covariance of each window's position with theta held fixed: H_xx^-1 (K, 3, 3)."""
    return laplace(dataclasses.replace(pb, free=np.zeros_like(pb.free)), x, theta)[0]


def mcmc_window(
    pb: Problem,
    k: int,
    x_map: FloatArray,
    theta: FloatArray,
    rng: np.random.Generator,
    walkers: int,
    steps: int,
) -> FloatArray:
    """emcee posterior samples of window k's position given theta (flat prior on x)."""
    import emcee

    one = pb.subset(np.arange(len(pb.t_obs)) == k)
    cinv = np.linalg.inv(pb.noise.covariance(k, pb.t_obs.shape[1]))

    def logp(xx: FloatArray) -> float:
        if xx[2] < -50.0:
            return -np.inf
        with np.errstate(invalid="ignore", divide="ignore"):
            r = one.evaluate(xx[None], theta)[0][0]
        return float(-0.5 * r @ cinv @ r) if np.all(np.isfinite(r)) else -np.inf

    chol = np.linalg.cholesky(conditional_cov(one, x_map[None], theta)[0] + 1e-9 * np.eye(3))
    p0 = x_map + rng.standard_normal((walkers, 3)) @ chol.T
    sampler = emcee.EnsembleSampler(walkers, 3, logp)
    sampler.random_state = np.random.RandomState(int(rng.integers(2**31))).get_state()  # no global RNG
    sampler.run_mcmc(p0, steps, progress=False)
    return sampler.get_chain(discard=steps // 2, flat=True)


# --- the method -----------------------------------------------------------------------


@dataclass
class _Windows:
    st: Setup
    fits: list[WindowFit]
    sel: np.ndarray  # indices of the windows used (coherent, placeable)
    t_obs: FloatArray
    noise: NoiseModel
    x0: FloatArray


def _windows(
    rec: Recording, mics: FloatArray, atmosphere: Atmosphere, cfg: ReconstructionConfig
) -> _Windows | None:
    st = prepare(rec, mics, atmosphere, cfg)
    fits = method_a_fits(st, cfg)
    if not fits:
        return None
    x0, _, ok0 = place_points(st, fits, rec.reported_t0, atmosphere)
    sel = np.flatnonzero(ok0 & np.array([f.quality >= cfg.min_peak for f in fits]))
    if not len(sel):
        return None
    t_obs = np.empty((len(sel), len(st.mics)))
    misfit = np.empty(len(sel))
    for i, j in enumerate(sel):
        d, misfit[i] = relative_delays(st, fits[j].tau, np.clip(fits[j].peak, 1e-3, None))
        t_obs[i] = fits[j].t_c + d - rec.reported_t0
    spread = np.array([fits[j].spread for j in sel])
    noise = window_noise(
        st.c, misfit, spread, len(st.pairs), len(st.mics), cfg, cfg.d_self_calibrate, cfg.d_self_calibrate
    )
    return _Windows(st, fits, sel, t_obs, noise, x0[sel])


def _problem(
    wins: list[_Windows], mics: FloatArray, atmosphere: Atmosphere, cfg: ReconstructionConfig
) -> Problem:
    n_groups, m = len(wins), len(mics)
    atm_mean, atm_std = prior_from_atmosphere(atmosphere, cfg)
    prior_mean = [atm_mean, np.zeros(n_groups)]
    prior_std = [atm_std, np.full(n_groups, cfg.d_sigma_t0_s)]
    model: Model
    if cfg.d_self_calibrate:  # medium, flash times and the array itself
        prior_mean += [np.zeros(m), np.zeros(3 * m)]
        prior_std += [
            np.full(m, max(cfg.d_sigma_clock_s, 1e-9)),
            np.full(3 * m, max(cfg.d_sigma_position_m, 1e-6)),
        ]

        def effective(
            xx: FloatArray, atm: FloatArray, dm: FloatArray
        ) -> tuple[FloatArray, FloatArray, FloatArray]:
            return effective_times(xx, mics + dm, atm)

        model = effective
    else:
        model = AtmosphereModel(atmosphere, mics)
    mean, std = np.concatenate(prior_mean), np.concatenate(prior_std)
    return Problem(
        t_obs=np.vstack([w.t_obs for w in wins]),
        noise=NoiseModel(
            np.concatenate([w.noise.a for w in wins]), np.concatenate([w.noise.b for w in wins])
        ),
        group=np.concatenate([np.full(len(w.sel), g) for g, w in enumerate(wins)]),
        model=model,
        prior_mean=mean,
        prior_std=std,
        free=np.full(len(mean), cfg.d_self_calibrate),
        array_cal=cfg.d_self_calibrate,
    )


def reconstruct_bayes_storm(
    recs: list[Recording], mic_positions: FloatArray, atmosphere: Atmosphere, cfg: ReconstructionConfig
) -> list[Reconstruction]:
    """Method D on several recordings of the same array and atmosphere (a storm): the medium
    parameters are shared, each recording keeps its own flash-time error."""
    mics = np.asarray(mic_positions, dtype=float)
    built = [_windows(r, mics, atmosphere, cfg) for r in recs]
    used = [i for i, w in enumerate(built) if w is not None]
    out = [Reconstruction.empty("D", cfg.config_hash()) for _ in recs]
    if not used:
        return out
    wins: list[_Windows] = [w for w in built if w is not None]
    pb = _problem(wins, wins[0].st.mics, atmosphere, cfg)
    x0 = np.vstack([w.x0 for w in wins])
    # One linear-algebra thread: multithreaded BLAS reorders floating-point sums, and the nearly
    # flat self-calibration posterior amplifies those last-digit differences into visibly
    # different answers. Single-threaded, the result is bit-for-bit reproducible.
    with threadpool_limits(limits=1):
        fit, inlier = infer(pb, x0, cfg)
    cov = fit.cov_x
    if cfg.d_inference == "mcmc":
        rng = np.random.default_rng(0)  # deterministic reference
        cov = cov.copy()
        cond = conditional_cov(pb, fit.x, fit.theta)
        for k in np.flatnonzero(inlier).tolist():
            samples = mcmc_window(pb, k, fit.x[k], fit.theta, rng, cfg.d_mcmc_walkers, cfg.d_mcmc_steps)
            # MCMC spread given theta, plus the Laplace term for the uncertainty of theta.
            cov[k] = np.cov(samples.T) + fit.cov_x[k] - cond[k]
    start = 0
    for g, (i, w) in enumerate(zip(used, wins, strict=True)):
        sl = slice(start, start + len(w.sel))
        start += len(w.sel)
        n = len(w.fits)
        x_all = np.full((n, 3), np.nan)
        cov_all = np.zeros((n, 3, 3))
        ok_all = np.zeros(n, dtype=bool)
        q_all = np.full(n, np.nan)
        x_all[w.sel], cov_all[w.sel], ok_all[w.sel], q_all[w.sel] = fit.x[sl], cov[sl], inlier[sl], fit.q[sl]
        fits_d = [dataclasses.replace(f, passed=bool(o)) for f, o in zip(w.fits, ok_all, strict=True)]
        keep = np.r_[np.arange(N_ATM), N_ATM + g]
        _, _, clock, dm = pb.split(fit.theta)
        extra = {
            "mic_clock_offsets_s": clock,
            "mic_position_offsets_m": dm,
            "theta": fit.theta[keep],  # (c0, c1, w0x, w0y, w1x, w1y, dt0 of this recording)
            "theta_cov": fit.cov_theta[np.ix_(keep, keep)],
            "theta_names": (*ATM_NAMES, "dt0"),
            "theta_prior_mean": pb.prior_mean[keep],
            "theta_prior_std": pb.prior_std[keep],
            "self_calibrated": bool(cfg.d_self_calibrate),
            "storm_recordings": len(wins),
            "window_q": q_all,
            "converged": fit.converged,
        }
        out[i] = assemble("D", w.st, fits_d, x_all, cov_all, ok_all, cfg, extra)
    return out


def reconstruct_bayes(
    rec: Recording, mic_positions: FloatArray, atmosphere: Atmosphere, cfg: ReconstructionConfig
) -> Reconstruction:
    """Method D on one recording (signals, nominal positions, reported t0, assumed atmosphere)."""
    return reconstruct_bayes_storm([rec], mic_positions, atmosphere, cfg)[0]
