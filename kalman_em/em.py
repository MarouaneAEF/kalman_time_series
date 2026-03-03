"""
EM algorithm for learning state-space model parameters.

E-step : Kalman filter + RTS smoother  →  sufficient statistics
M-step : closed-form parameter updates

References
----------
Shumway & Stoffer (1982), "An approach to time series smoothing and
forecasting using the EM algorithm", J. Time Series Analysis.
Ghahramani & Hinton (1996), "Parameter estimation for linear dynamical systems".
"""

import numpy as np
from .kalman_filter import kalman_filter, rts_smoother, lag_one_covariance


def e_step(Y, F, H, Q, R, mu0, Sigma0):
    """
    Run Kalman filter + RTS smoother and collect sufficient statistics.

    Returns
    -------
    stats : dict with keys
        'E_x'    (T, d)      E[x_t | Y]
        'E_xx'   (T, d, d)   E[x_t x_t^T | Y]
        'E_x1x'  (T-1, d, d) E[x_t x_{t-1}^T | Y]  (t=1..T-1)
        'log_lik' float
    """
    mu_filt, Sigma_filt, mu_pred, Sigma_pred, log_lik = kalman_filter(
        Y, F, H, Q, R, mu0, Sigma0
    )
    mu_smooth, Sigma_smooth, G = rts_smoother(
        mu_filt, Sigma_filt, mu_pred, Sigma_pred, F
    )
    C = lag_one_covariance(Sigma_smooth, G)   # C[t] = P_{t+1, t | T}

    E_xx  = Sigma_smooth + np.einsum("ti,tj->tij", mu_smooth, mu_smooth)
    E_x1x = C + np.einsum("ti,tj->tij", mu_smooth[1:], mu_smooth[:-1])

    return {
        "E_x":    mu_smooth,
        "E_xx":   E_xx,
        "E_x1x":  E_x1x,
        "log_lik": log_lik,
    }


def m_step(Y, stats, update_F=True, update_H=True,
           update_Q=True, update_R=True,
           diagonal_Q=False, diagonal_R=True):
    """Closed-form M-step updates."""
    T, m = Y.shape
    d = stats["E_x"].shape[1]
    E_x   = stats["E_x"]
    E_xx  = stats["E_xx"]
    E_x1x = stats["E_x1x"]

    # --- Initial state ---
    mu0    = E_x[0]
    Sigma0 = E_xx[0] - np.outer(E_x[0], E_x[0])
    Sigma0 = 0.5 * (Sigma0 + Sigma0.T)

    S11 = E_xx[1:].sum(axis=0)
    S10 = E_x1x.sum(axis=0)
    S00 = E_xx[:-1].sum(axis=0)

    # --- F ---
    if update_F:
        F = S10 @ np.linalg.inv(S00 + 1e-8 * np.eye(d))
    else:
        F = None

    # --- Q ---
    if update_Q:
        Q = (S11 - F @ S10.T) / (T - 1)
        Q = 0.5 * (Q + Q.T)
        if diagonal_Q:
            Q = np.diag(np.diag(Q))
    else:
        Q = None

    # --- H and R ---
    obs_mask = ~np.isnan(Y)
    S_yx = np.zeros((m, d))
    S_yy = np.zeros((m, m))
    S_xx_full = E_xx.sum(axis=0)

    for t in range(T):
        idx = obs_mask[t]
        if not idx.any():
            continue
        S_yx[idx] += np.outer(Y[t, idx], E_x[t])
        S_yy[np.ix_(idx, idx)] += np.outer(Y[t, idx], Y[t, idx])

    if update_H:
        H = S_yx @ np.linalg.inv(S_xx_full + 1e-8 * np.eye(d))
    else:
        H = None

    if update_R:
        R = (S_yy - H @ S_yx.T) / T
        R = 0.5 * (R + R.T)
        R = np.maximum(R, 1e-6 * np.eye(m))
        if diagonal_R:
            R = np.diag(np.diag(R))
    else:
        R = None

    return {"F": F, "H": H, "Q": Q, "R": R, "mu0": mu0, "Sigma0": Sigma0}


def _stable_init(Y, d, init_params, rng):
    """
    Build stable initial parameters.

    For d=2, m=1 (most common: stock price with level+trend),
    use a local linear trend structure.
    For other cases, use PCA-inspired init scaled to obs variance.
    """
    T, m = Y.shape
    obs_var = float(np.nanvar(Y))
    if obs_var == 0:
        obs_var = 1.0

    if d == 2 and m == 1 and "F" not in init_params:
        # Local linear trend: x_t = [level, trend], y_t = level + noise
        F  = np.array([[1.0, 1.0], [0.0, 1.0]])
        H  = np.array([[1.0, 0.0]])
        Q  = np.diag([obs_var * 0.05, obs_var * 0.001])
        R  = np.array([[obs_var * 0.3]])
        first_obs = Y[~np.isnan(Y[:, 0]), 0]
        mu0    = np.array([first_obs[0] if len(first_obs) else 0.0, 0.0])
        Sigma0 = np.diag([obs_var, obs_var * 0.01])
    else:
        F  = init_params.get("F",      np.eye(d) * 0.95)
        H  = init_params.get("H",      np.eye(m, d) * np.sqrt(obs_var))
        Q  = init_params.get("Q",      np.eye(d) * obs_var * 0.1)
        R  = init_params.get("R",      np.eye(m) * obs_var * 0.5)
        mu0    = init_params.get("mu0",    np.zeros(d))
        Sigma0 = init_params.get("Sigma0", np.eye(d) * obs_var)

    # Allow caller overrides
    F      = init_params.get("F",      F)
    H      = init_params.get("H",      H)
    Q      = init_params.get("Q",      Q)
    R      = init_params.get("R",      R)
    mu0    = init_params.get("mu0",    mu0)
    Sigma0 = init_params.get("Sigma0", Sigma0)

    return F, H, Q, R, mu0, Sigma0


def _clip_spectral_radius(F, max_rho=0.9999):
    """Scale F so its spectral radius does not exceed max_rho."""
    rho = np.max(np.abs(np.linalg.eigvals(F)))
    if rho > max_rho:
        F = F * (max_rho / rho)
    return F


def _ensure_pd(A, eps=1e-6):
    A = 0.5 * (A + A.T)
    min_eig = np.linalg.eigvalsh(A).min()
    if min_eig < eps:
        A += (eps - min_eig) * np.eye(A.shape[0])
    return A


def run_em(Y, d, n_iter=200, tol=1e-5,
           init_params=None,
           update_F=True, update_H=True,
           update_Q=True, update_R=True,
           diagonal_Q=False, diagonal_R=True,
           verbose=True):
    """
    Full EM loop for Kalman filter parameter estimation.

    Parameters
    ----------
    Y        : (T, m) observations (NaN = missing)
    d        : latent state dimension
    n_iter   : max EM iterations
    tol      : convergence tolerance on log-likelihood
    init_params : dict with optional keys F, H, Q, R, mu0, Sigma0
    verbose  : print progress every 10 iters

    Returns
    -------
    params   : dict {F, H, Q, R, mu0, Sigma0}
    log_liks : list of log-likelihoods
    """
    if init_params is None:
        init_params = {}

    rng = np.random.default_rng(0)
    F, H, Q, R, mu0, Sigma0 = _stable_init(Y, d, init_params, rng)

    log_liks = []

    for i in range(n_iter):
        stats = e_step(Y, F, H, Q, R, mu0, Sigma0)
        ll = stats["log_lik"]
        log_liks.append(ll)

        if verbose and (i % 10 == 0 or i == n_iter - 1):
            print(f"  EM iter {i:4d} | log-lik = {ll:.4f}")

        if np.isnan(ll):
            print("  Warning: log-likelihood is NaN — stopping early.")
            break

        if i > 0 and abs(log_liks[-1] - log_liks[-2]) < tol:
            if verbose:
                print(f"  Converged at iteration {i}.")
            break

        new = m_step(Y, stats,
                     update_F=update_F, update_H=update_H,
                     update_Q=update_Q, update_R=update_R,
                     diagonal_Q=diagonal_Q, diagonal_R=diagonal_R)

        if update_F:
            F = _clip_spectral_radius(new["F"])
        if update_H:
            H = new["H"]
        if update_Q:
            Q = _ensure_pd(new["Q"])
        if update_R:
            R = _ensure_pd(new["R"])
        mu0    = new["mu0"]
        Sigma0 = _ensure_pd(new["Sigma0"])

    return {"F": F, "H": H, "Q": Q, "R": R, "mu0": mu0, "Sigma0": Sigma0}, log_liks
