"""Unit tests for kalman_em.model.KalmanEM (public API)."""
import numpy as np
import pytest
from kalman_em import KalmanEM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_series(n=80, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * t / 12) + 0.1 * rng.standard_normal(n)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

class TestFit:

    def test_fit_sets_params(self):
        Y = make_series()
        model = KalmanEM(d=2, n_iter=20, verbose=False).fit(Y)
        assert model.params_ is not None
        assert model.log_liks_ is not None

    def test_fit_1d_input(self):
        Y = make_series()
        model = KalmanEM(d=2, n_iter=20, verbose=False).fit(Y)
        assert model.params_["H"].shape[1] == 2

    def test_fit_2d_input(self):
        Y = make_series().reshape(-1, 1)
        model = KalmanEM(d=2, n_iter=20, verbose=False).fit(Y)
        assert model.params_ is not None

    def test_fit_returns_self(self):
        Y = make_series()
        model = KalmanEM(d=2, n_iter=10, verbose=False)
        assert model.fit(Y) is model

    def test_fit_with_missing_data(self):
        Y = make_series()
        Y[5:10] = np.nan
        model = KalmanEM(d=2, n_iter=15, verbose=False).fit(Y)
        assert model.params_ is not None

    def test_log_liks_improves(self):
        Y = make_series()
        model = KalmanEM(d=2, n_iter=30, verbose=False).fit(Y)
        # Early iterations must show clear improvement
        mid = max(2, len(model.log_liks_) // 2)
        diffs = np.diff(model.log_liks_[:mid])
        assert np.all(diffs >= -1e-4), f"Log-liks decreased early: {diffs.min():.6f}"
        # Overall log-likelihood must improve from start to finish
        assert model.log_liks_[-1] > model.log_liks_[0]


# ---------------------------------------------------------------------------
# filter / smooth
# ---------------------------------------------------------------------------

class TestFilterSmooth:

    @pytest.fixture(autouse=True)
    def fitted_model(self):
        Y = make_series()
        self.Y = Y
        self.model = KalmanEM(d=2, n_iter=20, verbose=False).fit(Y)

    def test_filter_output_shapes(self):
        mu, var = self.model.filter(self.Y)
        assert mu.shape == self.Y.reshape(-1, 1).shape
        assert var.shape == self.Y.reshape(-1, 1).shape

    def test_filter_variances_positive(self):
        _, var = self.model.filter(self.Y)
        assert np.all(var > 0)

    def test_smooth_output_shapes(self):
        mu, var = self.model.smooth(self.Y)
        assert mu.shape == self.Y.reshape(-1, 1).shape
        assert var.shape == self.Y.reshape(-1, 1).shape

    def test_smooth_var_leq_filter_var(self):
        """Smoothed uncertainty must be <= filtered uncertainty."""
        _, var_f = self.model.filter(self.Y)
        _, var_s = self.model.smooth(self.Y)
        assert np.all(var_s <= var_f + 1e-8), (
            "Smoothed variance exceeds filtered variance"
        )


# ---------------------------------------------------------------------------
# predict_one_step
# ---------------------------------------------------------------------------

class TestPredictOneStep:

    @pytest.fixture(autouse=True)
    def fitted_model(self):
        Y = make_series(n=100)
        self.Y_train = Y[:80]
        self.Y_test = Y[80:]
        self.model = KalmanEM(d=2, n_iter=20, verbose=False).fit(self.Y_train)

    def test_output_shapes(self):
        y_pred, y_var = self.model.predict_one_step(self.Y_test)
        assert y_pred.shape[0] == len(self.Y_test)
        assert y_var.shape == y_pred.shape

    def test_variances_are_positive(self):
        _, y_var = self.model.predict_one_step(self.Y_test)
        assert np.all(y_var > 0)

    def test_with_context(self):
        y_pred, y_var = self.model.predict_one_step(
            self.Y_test, Y_context=self.Y_train
        )
        assert y_pred.shape[0] == len(self.Y_test)
        assert np.all(y_var > 0)


# ---------------------------------------------------------------------------
# forecast
# ---------------------------------------------------------------------------

class TestForecast:

    @pytest.fixture(autouse=True)
    def fitted_model(self):
        Y = make_series()
        self.Y = Y
        self.model = KalmanEM(d=2, n_iter=20, verbose=False).fit(Y)

    def test_output_shapes(self):
        n = 12
        y_fc, y_var = self.model.forecast(self.Y, n_steps=n)
        assert y_fc.shape[0] == n
        assert y_var.shape == y_fc.shape

    def test_variances_are_positive(self):
        _, y_var = self.model.forecast(self.Y, n_steps=10)
        assert np.all(y_var > 0)

    def test_uncertainty_grows_with_horizon(self):
        """Forecast variance should be non-decreasing over the horizon."""
        _, y_var = self.model.forecast(self.Y, n_steps=20)
        diffs = np.diff(y_var.ravel())
        assert np.all(diffs >= -1e-8), (
            "Forecast variance decreased over horizon"
        )
