"""Shared fixtures for the test suite."""
import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_params():
    """Minimal 1-D state, 1-D observation Kalman parameters."""
    d, m = 2, 1
    F = np.array([[1.0, 1.0], [0.0, 0.9]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(d) * 0.1
    R = np.eye(m) * 0.5
    mu0 = np.zeros(d)
    Sigma0 = np.eye(d)
    return dict(F=F, H=H, Q=Q, R=R, mu0=mu0, Sigma0=Sigma0)


@pytest.fixture
def synthetic_data(simple_params, rng):
    """100 observations from a local-linear-trend model."""
    T = 100
    F, H, Q, R = (simple_params[k] for k in ("F", "H", "Q", "R"))
    mu0, Sigma0 = simple_params["mu0"], simple_params["Sigma0"]
    d = F.shape[0]

    x = rng.multivariate_normal(mu0, Sigma0)
    Y = np.empty((T, H.shape[0]))
    for t in range(T):
        Y[t] = H @ x + rng.multivariate_normal(np.zeros(H.shape[0]), R)
        x = F @ x + rng.multivariate_normal(np.zeros(d), Q)
    return Y


