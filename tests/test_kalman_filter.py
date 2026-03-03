"""Unit tests for kalman_em.kalman_filter."""
import numpy as np
import pytest
from kalman_em.kalman_filter import kalman_filter, rts_smoother, lag_one_covariance
from tests.utils import is_symmetric, is_psd


# ---------------------------------------------------------------------------
# kalman_filter
# ---------------------------------------------------------------------------

class TestKalmanFilter:

    def test_output_shapes(self, synthetic_data, simple_params):
        Y = synthetic_data
        T, m = Y.shape
        d = simple_params["F"].shape[0]
        mu_f, Sig_f, mu_p, Sig_p, ll = kalman_filter(Y, **simple_params)

        assert mu_f.shape == (T, d)
        assert Sig_f.shape == (T, d, d)
        assert mu_p.shape == (T, d)
        assert Sig_p.shape == (T, d, d)
        assert np.isscalar(ll) or ll.ndim == 0

    def test_covariances_are_symmetric_psd(self, synthetic_data, simple_params):
        Y = synthetic_data
        _, Sig_f, _, Sig_p, _ = kalman_filter(Y, **simple_params)

        for t in range(len(Y)):
            assert is_symmetric(Sig_f[t]), f"Sigma_filt[{t}] not symmetric"
            assert is_psd(Sig_f[t]),       f"Sigma_filt[{t}] not PSD"
            assert is_symmetric(Sig_p[t]), f"Sigma_pred[{t}] not symmetric"
            assert is_psd(Sig_p[t]),       f"Sigma_pred[{t}] not PSD"

    def test_log_likelihood_is_finite(self, synthetic_data, simple_params):
        Y = synthetic_data
        *_, ll = kalman_filter(Y, **simple_params)
        assert np.isfinite(ll)

    def test_missing_data_handled(self, synthetic_data, simple_params):
        """NaN rows should not raise and should not propagate into state means."""
        Y = synthetic_data.copy()
        Y[10] = np.nan
        Y[50] = np.nan
        mu_f, _, _, _, ll = kalman_filter(Y, **simple_params)
        assert np.all(np.isfinite(mu_f))
        assert np.isfinite(ll)

    def test_identity_transition_preserves_mean(self, rng):
        """With F=I, Q=0, the filtered mean should equal mu0 for all-NaN obs."""
        d, m, T = 2, 1, 10
        F = np.eye(d)
        H = np.array([[1.0, 0.0]])
        Q = np.zeros((d, d))
        R = np.eye(m)
        mu0 = np.array([3.0, 1.0])
        Sigma0 = np.zeros((d, d))  # known initial state
        Y = np.full((T, m), np.nan)  # all missing

        mu_f, _, _, _, _ = kalman_filter(Y, F, H, Q, R, mu0, Sigma0)
        np.testing.assert_allclose(mu_f, np.tile(mu0, (T, 1)), atol=1e-10)


# ---------------------------------------------------------------------------
# rts_smoother
# ---------------------------------------------------------------------------

class TestRtsSmoother:

    def _run(self, Y, params):
        mu_f, Sig_f, mu_p, Sig_p, _ = kalman_filter(Y, **params)
        mu_s, Sig_s, G = rts_smoother(mu_f, Sig_f, mu_p, Sig_p, params["F"])
        return mu_f, Sig_f, mu_s, Sig_s, G

    def test_output_shapes(self, synthetic_data, simple_params):
        Y = synthetic_data
        T, _ = Y.shape
        d = simple_params["F"].shape[0]
        mu_f, Sig_f, mu_s, Sig_s, G = self._run(Y, simple_params)

        assert mu_s.shape == (T, d)
        assert Sig_s.shape == (T, d, d)
        assert G.shape == (T - 1, d, d)

    def test_last_step_equals_filter(self, synthetic_data, simple_params):
        """At t=T-1, smoother must equal filter (no future data)."""
        Y = synthetic_data
        mu_f, Sig_f, mu_s, Sig_s, _ = self._run(Y, simple_params)

        np.testing.assert_allclose(mu_s[-1], mu_f[-1], atol=1e-10)
        np.testing.assert_allclose(Sig_s[-1], Sig_f[-1], atol=1e-10)

    def test_smoothing_reduces_uncertainty(self, synthetic_data, simple_params):
        """Smoothed variance <= filtered variance at every time step."""
        Y = synthetic_data
        _, Sig_f, _, Sig_s, _ = self._run(Y, simple_params)

        for t in range(len(Y) - 1):
            diff = Sig_f[t] - Sig_s[t]
            assert is_psd(diff), (
                f"Sigma_filt[{t}] - Sigma_smooth[{t}] is not PSD "
                f"(smoothing increased uncertainty)"
            )

    def test_smoothed_covariances_are_symmetric_psd(self, synthetic_data, simple_params):
        Y = synthetic_data
        _, _, _, Sig_s, _ = self._run(Y, simple_params)

        for t in range(len(Y)):
            assert is_symmetric(Sig_s[t]), f"Sigma_smooth[{t}] not symmetric"
            assert is_psd(Sig_s[t]),       f"Sigma_smooth[{t}] not PSD"


# ---------------------------------------------------------------------------
# lag_one_covariance
# ---------------------------------------------------------------------------

class TestLagOneCovariance:

    def test_output_shape(self, synthetic_data, simple_params):
        Y = synthetic_data
        T, d = Y.shape[0], simple_params["F"].shape[0]
        mu_f, Sig_f, mu_p, Sig_p, _ = kalman_filter(Y, **simple_params)
        mu_s, Sig_s, G = rts_smoother(mu_f, Sig_f, mu_p, Sig_p, simple_params["F"])

        C = lag_one_covariance(Sig_s, G)
        assert C.shape == (T - 1, d, d)
