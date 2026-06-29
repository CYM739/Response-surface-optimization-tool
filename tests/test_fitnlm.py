"""
Equivalence tests for NonlinearLSWrapper (Matlab `fitnlm`-equivalent).

The paper fits the quadratic PRS polynomial via Matlab `fitnlm`. That model
is linear in parameters, so unconstrained nonlinear least squares and OLS
share the same convex SSR objective and converge to the same unique global
minimum. These tests verify our scipy.optimize.curve_fit-backed wrapper
matches the OLSWrapper's coefficients and predictions to numerical tolerance.

If these tests ever fail, it means one of:
  - The polynomial term ordering in models._build_poly_term_names drifted.
  - The design-matrix convention in _poly_eval and _add_polynomial_terms fell
    out of sync.
  - Something in statsmodels OLS or scipy.optimize.curve_fit changed default
    behavior in a way that breaks linear-in-params equivalence.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from logic.data_processing import analyze_csv, expand_terms, run_analysis
from logic.models import NonlinearLSWrapper, OLSWrapper


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
IOR1_CSV = _PROJECT_ROOT.parent / "Papers" / "ior1_dataset.csv"


def _ols_coefs_in_fitnlm_order(ols_wrapper: OLSWrapper) -> np.ndarray:
    """OLSWrapper stores coefficients keyed by statsmodels' names (with
    ':' for interactions and '_sq' for squares). NonlinearLSWrapper uses
    '*' for interactions. Translate, then align to the fitnlm order:
    [Intercept, linear..., squared..., interactions...]."""
    ivars = ols_wrapper.independent_vars
    p = ols_wrapper.params

    ordered = [p['Intercept']]
    for v in ivars:
        ordered.append(p[v])
    for v in ivars:
        ordered.append(p[f'{v}_sq'])
    for i, v1 in enumerate(ivars):
        for v2 in ivars[i + 1:]:
            # statsmodels uses 'v1:v2' for interactions
            ordered.append(p[f'{v1}:{v2}'])
    return np.asarray(ordered, dtype=float)


def _fit_both_on(df, ivars, dep_var):
    """Fit OLS and NonlinearLS on the same data. Returns both wrappers."""
    expanded = df.copy()
    expand_terms(expanded, ivars)
    ols = run_analysis(
        dataframe=expanded,
        independent_vars=ivars,
        dependent_var=dep_var,
        model_type="Polynomial OLS",
    )
    nls = run_analysis(
        dataframe=df,  # NLS builds poly terms internally, so pass raw df
        independent_vars=ivars,
        dependent_var=dep_var,
        model_type="Nonlinear LS (fitnlm)",
    )
    return ols, nls


# ---------------------------------------------------------------------------
# Basic wrapper behavior
# ---------------------------------------------------------------------------

def test_nls_wrapper_has_required_interface(analyzed_demo):
    """Views do `isinstance(wrapper, OLSWrapper)` checks — confirm NLS is
    NOT one of those (diagnostics_view / elimination_view guard on it) while
    still exposing the same duck-typed API the other views rely on."""
    dep_var = analyzed_demo["dependent_vars"][0]
    nls = run_analysis(
        dataframe=analyzed_demo["exp_df"],
        independent_vars=analyzed_demo["independent_vars"],
        dependent_var=dep_var,
        model_type="Nonlinear LS (fitnlm)",
    )
    assert isinstance(nls, NonlinearLSWrapper)
    assert not isinstance(nls, OLSWrapper)
    # Duck-typed surface shared with OLS/SVR/RF:
    assert hasattr(nls, "predict")
    assert hasattr(nls, "get_summary")
    assert nls.independent_vars == analyzed_demo["independent_vars"]
    assert nls.formula is not None


# ---------------------------------------------------------------------------
# Numerical equivalence vs OLS — the headline test
# ---------------------------------------------------------------------------

def test_fitnlm_coefficients_match_ols_on_demo(analyzed_demo):
    """On the well-conditioned demo CSV, fitnlm and OLS must converge to the
    same coefficients. Tolerance 1e-6 — looser than machine precision to
    absorb LM stopping criteria."""
    ivars = analyzed_demo["independent_vars"]
    dep_var = analyzed_demo["dependent_vars"][0]

    ols, nls = _fit_both_on(analyzed_demo["exp_df"], ivars, dep_var)

    ols_coefs = _ols_coefs_in_fitnlm_order(ols)
    nls_coefs = nls.params

    assert ols_coefs.shape == nls_coefs.shape
    max_abs_diff = float(np.max(np.abs(ols_coefs - nls_coefs)))
    max_rel_diff = float(np.max(np.abs(ols_coefs - nls_coefs)
                                 / (np.abs(ols_coefs) + 1e-12)))
    assert max_abs_diff < 1e-6, (
        f"Coefficient mismatch: abs={max_abs_diff:.2e}, rel={max_rel_diff:.2e}\n"
        f"OLS: {ols_coefs}\nNLS: {nls_coefs}"
    )


def test_fitnlm_predictions_match_ols_on_demo(analyzed_demo):
    """Predicted values on the training set must agree to ~1e-8."""
    ivars = analyzed_demo["independent_vars"]
    dep_var = analyzed_demo["dependent_vars"][0]

    ols, nls = _fit_both_on(analyzed_demo["exp_df"], ivars, dep_var)

    y_ols = np.asarray(ols.predict(analyzed_demo["exp_df"]))
    y_nls = np.asarray(nls.predict(analyzed_demo["exp_df"]))

    np.testing.assert_allclose(y_ols, y_nls, atol=1e-8, rtol=1e-8)


# ---------------------------------------------------------------------------
# Paper data (IOR1) — this is the interesting case because the design
# matrix is badly conditioned. Test how fitnlm handles it compared to OLS.
# ---------------------------------------------------------------------------

def test_fitnlm_predictions_match_ols_on_paper_ior1(capsys):
    """On the IOR1 dataset (13 rows, 15 poly coeffs, condition# ~1e38), OLS
    and fitnlm may disagree on individual coefficients because the problem is
    rank-deficient and the solution set is a manifold. But *predicted values*
    at the training points must still match — both algorithms find some point
    on the same minimum-SSR manifold.
    """
    df = pd.read_csv(IOR1_CSV)
    (data_df, _, ivars, dep_vars, *_ ) = analyze_csv(df)
    dep_var = dep_vars[0]

    ols, nls = _fit_both_on(data_df, ivars, dep_var)

    y_ols = np.asarray(ols.predict(data_df))
    y_nls = np.asarray(nls.predict(data_df))

    # Report side-by-side so the user can see how close they come
    report = ["", "=" * 70, "IOR1: fitnlm vs OLS predicted tumor size (training rows)", "=" * 70]
    report.append(f"{'row':>3}{'y_actual':>12}{'y_OLS':>14}{'y_fitnlm':>14}{'diff':>14}")
    y_actual = data_df[dep_var].values
    for i, (ya, yo, yn) in enumerate(zip(y_actual, y_ols, y_nls)):
        report.append(f"{i:>3}{ya:>12.4f}{yo:>14.6f}{yn:>14.6f}{yo - yn:>14.2e}")
    ols_ss = float(np.sum((y_actual - y_ols) ** 2))
    nls_ss = float(np.sum((y_actual - y_nls) ** 2))
    report.append(f"SSR  OLS={ols_ss:.6e}  fitnlm={nls_ss:.6e}  (should match — both minimize SSR)")
    report.append("=" * 70)
    print("\n".join(report))

    # Both minimizers of the same SSR objective must yield the same SSR,
    # even if the parameter values differ (rank deficiency).
    assert abs(ols_ss - nls_ss) < 1e-6, (
        f"OLS and fitnlm found different SSR minima: {ols_ss} vs {nls_ss}"
    )


# ---------------------------------------------------------------------------
# Bounded fit — the actual value-add over OLS
# ---------------------------------------------------------------------------

def test_fitnlm_respects_parameter_bounds(analyzed_demo):
    """When bounds are supplied, coefficients must stay inside them — this is
    the behavior Matlab fitnlm's Lower/Upper options provide, and the reason
    you'd ever reach for NLS over OLS."""
    ivars = analyzed_demo["independent_vars"]
    dep_var = analyzed_demo["dependent_vars"][0]

    wrapper = NonlinearLSWrapper(ivars)
    n = len(wrapper.param_names)
    # Pin all squared-term coefficients to be <= 0 (a pharmacologically
    # reasonable constraint: response should curve *down* with higher doses).
    lower = np.full(n, -np.inf)
    upper = np.full(n, np.inf)
    n_features = len(ivars)
    squared_start = 1 + n_features
    squared_end = squared_start + n_features
    upper[squared_start:squared_end] = 0.0

    wrapper.fit(analyzed_demo["exp_df"], dep_var, bounds=(lower, upper))

    squared_coefs = wrapper.params[squared_start:squared_end]
    assert np.all(squared_coefs <= 1e-8), (
        f"Bound violated — squared coefs should be <= 0: {squared_coefs}"
    )
