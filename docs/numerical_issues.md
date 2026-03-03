# Numerical issues encountered and resolutions

> Summary of numerical instabilities observed during the implementation of the Kalman filter with EM algorithm, and the fixes applied.

---

## 1. Out-of-scope variable in the Kalman filter

### Symptom
```
UnboundLocalError: local variable 'obs_idx' referenced before assignment
```

### Cause
In `kalman_filter.py`, the variable `obs_idx` (boolean mask of non-missing observations)
was only defined inside a conditional block `if missing.any()`, but was used later in the
log-likelihood computation — even when there were no missing values.

```python
# BEFORE — obs_idx undefined if missing.any() == False
if missing.any():
    obs_idx = ~missing
    ...
# later:
ll += -0.5 * (y_obs @ np.linalg.solve(S[obs_idx], y_obs))  # ← crash
```

### Fix
Define `obs_idx` unconditionally before any branching:

```python
obs_idx = ~missing          # always defined
if missing.any():
    H_t = H[obs_idx]
    ...
```

**Lesson**: never conditionally define a variable that downstream code depends on.

---

## 2. Log-likelihood explosion after the first EM iteration

### Symptom
```
iter   1 | ll = -1274.38
iter   2 | ll =     nan
iter   3 | ll =     nan
```
All predictions become `nan` from the second iteration onward.

### Root cause (causal chain)

| Step | What happens |
|---|---|
| Initialisation | `H` initialised randomly at small scale (`× 0.1`) |
| Blind filter | With a near-zero `H`, the filter ignores observations → `mu_filt ≈ mu_pred` |
| M-step F | Smoothed statistics produce a matrix `F` with spectral radius `> 1` |
| Next iteration | Predicted states diverge exponentially → covariances `→ ∞` → `nan` |

### Fixes applied

**a) Smart initialisation (Local Linear Trend)**

For the `d=2, m=1` case (most common), replace random init with a known
local trend model structure:

```python
F  = np.array([[1.0, 1.0], [0.0, 1.0]])   # level + velocity
H  = np.array([[1.0, 0.0]])                # we observe the position
Q  = np.diag([obs_var * 0.05, obs_var * 0.001])
R  = np.array([[obs_var * 0.3]])
mu0    = np.array([y[0], 0.0])
Sigma0 = np.diag([obs_var, obs_var * 0.01])
```

This guarantees that the filter "sees" the observations from the very first step.

**b) Spectral radius clipping of F after each M-step**

```python
def _clip_spectral_radius(F, max_rho=0.9999):
    eigvals = np.linalg.eigvals(F)
    rho = np.max(np.abs(eigvals))
    if rho > max_rho:
        F = F * (max_rho / rho)
    return F
```

Prevents `F` from producing explosive dynamics regardless of data quality.

**c) Joseph form for the covariance update**

Replace the standard form (numerically fragile):
```python
# Standard — may produce a non-positive-definite matrix if K is ill-conditioned
P_new = (I - K @ H) @ P_pred
```
with the symmetric Joseph form:
```python
# Joseph — guarantees symmetry and positive semi-definiteness
ImKH = I - K @ H
P_new = ImKH @ P_pred @ ImKH.T + K @ R @ K.T
```

**d) Systematic regularisation of covariance matrices**

After each update, enforce positive definiteness:
```python
def _make_pd(A, eps=1e-6):
    A = 0.5 * (A + A.T)                        # exact symmetry
    w, v = np.linalg.eigh(A)
    w = np.maximum(w, eps)                      # eigenvalues ≥ ε
    return v @ np.diag(w) @ v.T
```

Applied to `Q`, `R`, `Sigma0` and `P` at every step.

---

## 2. Error in the lag-one covariance computation

### Symptom
Very slow or oscillating EM convergence, log-likelihood stagnating or increasing.

### Cause
The cross-covariance `C_t = E[x_{t+1} x_t^T | Y]` (required by the M-step to estimate `F`)
was computed via a complex backward recursion, which turned out to be incorrect in cases
with missing observations.

### Fix
Use the direct formula from Ghahramani & Hinton (1996):

```python
# G[t] = smoother gain at step t
# Sigma_smooth[t] = smoothed covariance at step t

C[t] = Sigma_smooth[t+1] @ G[t].T

# Vectorised over all T:
C = np.einsum('tij,tkj->tik', Sigma_smooth[1:], G)
```

This formula is direct, vectorisable, and does not propagate numerical errors.

---

## 3. Non-convergence on strongly seasonal data

### Symptom
On French electricity consumption data (annual seasonality ~80 GW in winter
vs ~36 GW in summer), the raw Kalman filter `d=2` stagnates at **MAPE ≈ 5 %** after
200 iterations without ever converging.

### Cause
The `d=2` linear state model is structurally unable to represent a 365-day seasonality
with a single trend component. The EM compensates by inflating `Q` (process noise),
making the filter too "agile" and unstable.

### Fix: STL + Kalman-EM pipeline

Decompose the series before running the EM:

```
Raw series
    │
    ▼
STL (Seasonal-Trend via Loess, period=365)
    ├── trend      → linear extrapolation for forecast
    ├── seasonal   → repeated from last year's profile
    └── residual (16 % of total variance)
                │
                ▼
           Kalman-EM  (d=2, 200 iters)
                │
                ▼
        residual forecast
                │
                ▼
    Recomposition: residual_forecast + seasonal + trend
```

**Result**: MAPE drops from **5.0 %** to **1.88 %**.

The same approach on SNCF TGV data (monthly, period=12) yields:
- `trains_prevus`: MAPE = 2.49 %
- `ponctualite_pct`: MAPE = 1.01 %

---

## 4. Pandas deprecation warning on Series indexing

### Symptom
```
FutureWarning: Series.__getitem__ treating keys as positions is deprecated.
```

### Cause
Accessing `trend[-1]` on a `pd.Series` with a non-standard integer index — Python
interprets `-1` as a relative position, which pandas is deprecating.

### Fix
```python
# BEFORE
last_trend = trend[-1]

# AFTER
last_trend = trend.iloc[-1]
```

General rule: always use `.iloc[]` for positional indexing and `.loc[]` for
label-based indexing on pandas objects.

---

## Conclusion: best practices from this implementation

### 1. Always initialise with a physically coherent model
Random initialisation is dangerous for state-space models: a near-zero `H` blinds the
filter, and the EM cannot recover on its own. **Starting from a known structure**
(Local Linear Trend, Random Walk + drift, etc.) dramatically improves convergence.

### 2. Control the spectral radius of the transition matrix
After each M-step, verify that `ρ(F) ≤ 1`. A model with `ρ(F) > 1` is explosive by
construction. Clipping is a cheap guard that prevents catastrophic divergence.

### 3. Use the Joseph form for covariance updates
The form `P = (I-KH)P_pred` is numerically unstable. The Joseph form
`P = (I-KH)P_pred(I-KH)^T + KRK^T` guarantees symmetry and positive semi-definiteness
at the cost of a marginally more expensive computation.

### 4. Regularise all covariance matrices
After every operation (M-step, update, smooth), apply `_make_pd` to `Q`, `R`, `Sigma0`
and all `P_t`. A regularisation of `eps = 1e-6` is invisible on real data but
prevents numerical singularities.

### 5. Prefer direct formulas over complex recursions
The lag-one covariance `C_t = Sigma_{t+1|T} G_t^T` is direct and vectorisable. Backward
recursions propagate errors and are harder to debug. In general,
**the closed-form solution is more reliable than the equivalent recursion**.

### 6. Decompose before modelling seasonal series
A low-dimensional linear Kalman filter cannot capture strong seasonality without
over-parameterisation. The **STL → Kalman-EM on residual → recomposition** pipeline
is more robust and interpretable than increasing the state dimension to absorb seasonality.

### 7. Pandas indexing: `.iloc[]` and `.loc[]` unambiguously
Never use `series[-1]` or `series[n]` directly. Always use `.iloc[-1]` (position) or
`.loc[label]` (label) to avoid deprecation warnings and ambiguous behaviour.

---

*Document generated from the implementation experience of the `kalman_time_series` project.*
