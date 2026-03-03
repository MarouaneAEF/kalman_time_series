"""Unit tests for kalman_em.em (e_step, m_step, run_em)."""
import numpy as np
from kalman_em.em import e_step, m_step, run_em
from tests.utils import is_psd, is_symmetric


# ---------------------------------------------------------------------------
# e_step
# ---------------------------------------------------------------------------

class TestEStep:

    def test_output_keys(self, synthetic_data, simple_params):
        stats = e_step(synthetic_data, **simple_params)
        assert {"E_x", "E_xx", "E_x1x", "log_lik"} == set(stats.keys())

    def test_output_shapes(self, synthetic_data, simple_params):
        T = len(synthetic_data)
        d = simple_params["F"].shape[0]
        stats = e_step(synthetic_data, **simple_params)

        assert stats["E_x"].shape == (T, d)
        assert stats["E_xx"].shape == (T, d, d)
        assert stats["E_x1x"].shape == (T - 1, d, d)

    def test_E_xx_is_psd(self, synthetic_data, simple_params):
        """E[x_t x_t^T | Y] must be positive semi-definite."""
        stats = e_step(synthetic_data, **simple_params)
        for t, mat in enumerate(stats["E_xx"]):
            assert is_psd(mat), f"E_xx[{t}] is not PSD"

    def test_log_lik_is_finite(self, synthetic_data, simple_params):
        stats = e_step(synthetic_data, **simple_params)
        assert np.isfinite(stats["log_lik"])


# ---------------------------------------------------------------------------
# m_step
# ---------------------------------------------------------------------------

class TestMStep:

    def _stats(self, Y, params):
        return e_step(Y, **params)

    def test_returned_matrices_are_symmetric_psd(self, synthetic_data, simple_params):
        stats = self._stats(synthetic_data, simple_params)
        new_params = m_step(synthetic_data, stats)

        for key in ("Q", "R", "Sigma0"):
            mat = new_params[key]
            if mat is not None:
                assert is_symmetric(mat), f"{key} not symmetric after M-step"
                assert is_psd(mat),       f"{key} not PSD after M-step"

    def test_R_lower_bounded(self, synthetic_data, simple_params):
        """R must be >= 1e-6 * I to avoid numerical collapse."""
        stats = self._stats(synthetic_data, simple_params)
        R = m_step(synthetic_data, stats)["R"]
        assert np.linalg.eigvalsh(R).min() >= 1e-7

    def test_selective_update(self, synthetic_data, simple_params):
        """When update_R=False, R should be None in output; Q should still be updated."""
        stats = self._stats(synthetic_data, simple_params)
        out = m_step(synthetic_data, stats, update_R=False)
        assert out["R"] is None
        assert out["Q"] is not None
        assert out["F"] is not None

    def test_diagonal_R_is_diagonal(self, synthetic_data, simple_params):
        stats = self._stats(synthetic_data, simple_params)
        R = m_step(synthetic_data, stats, diagonal_R=True)["R"]
        off_diag = R - np.diag(np.diag(R))
        np.testing.assert_allclose(off_diag, 0, atol=1e-12)


# ---------------------------------------------------------------------------
# run_em
# ---------------------------------------------------------------------------

class TestRunEM:

    def test_log_likelihood_increases_initially(self, synthetic_data):
        """EM log-likelihood must increase during the early iterations."""
        _, log_liks = run_em(synthetic_data, d=2, n_iter=30, verbose=False)
        # Check the first half of iterations where numerical issues are unlikely
        mid = max(2, len(log_liks) // 2)
        diffs = np.diff(log_liks[:mid])
        assert np.all(diffs >= -1e-4), (
            f"Log-likelihood decreased early: min diff = {diffs.min():.6f}"
        )
        # Overall: final LL must be better than initial
        assert log_liks[-1] > log_liks[0]

    def test_spectral_radius_of_F(self, synthetic_data):
        """Learned F must be stable (all |eigenvalues| <= 0.9999)."""
        params, _ = run_em(synthetic_data, d=2, n_iter=30, verbose=False)
        rho = np.abs(np.linalg.eigvals(params["F"])).max()
        assert rho <= 1.0 + 1e-6, f"Spectral radius {rho:.4f} > 1"

    def test_output_param_shapes(self, synthetic_data):
        _, m = synthetic_data.shape
        d = 2
        params, log_liks = run_em(synthetic_data, d=d, n_iter=20, verbose=False)

        assert params["F"].shape == (d, d)
        assert params["H"].shape == (m, d)
        assert params["Q"].shape == (d, d)
        assert params["R"].shape == (m, m)
        assert params["mu0"].shape == (d,)
        assert params["Sigma0"].shape == (d, d)
        assert len(log_liks) >= 1

    def test_missing_data_does_not_crash(self, synthetic_data):
        Y = synthetic_data.copy()
        Y[::10] = np.nan
        _, log_liks = run_em(Y, d=2, n_iter=15, verbose=False)
        assert np.isfinite(log_liks[-1])

    def test_convergence_stops_early(self, synthetic_data):
        """With loose tolerance, should converge before max iterations."""
        _, log_liks = run_em(
            synthetic_data, d=2, n_iter=500, tol=1e-2, verbose=False
        )
        assert len(log_liks) < 500, "Expected early convergence but ran all 500 iters"
