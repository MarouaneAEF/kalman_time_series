"""Shared test utilities."""
import numpy as np


def is_symmetric(A, atol=1e-8):
    return np.allclose(A, A.T, atol=atol)


def is_psd(A, tol=1e-8):
    return bool(np.linalg.eigvalsh(A).min() >= -tol)
