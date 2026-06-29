"""
Step 1 tests: scale_inputs=True on NonlinearLSWrapper.

Goal is to verify that standardizing inputs:
  1. Leaves predictions invariant (fits the same response surface in
     raw-input space, just with better conditioning internally).
  2. Drops the design-matrix condition number by many orders of magnitude
     on the IOR1 paper data (the whole reason we added it).
  3. Keeps the predict(df) interface unchanged — callers pass raw units
     and get raw-unit predictions back.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from logic.data_processing import analyze_csv, expand_terms, run_analysis
from logic.helpers import _add_polynomial_terms
from logic.interpolation import interpolate_to_n_rows


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
IOR1_CSV = _PROJECT_ROOT.parent / "Papers" / "ior1_dataset.csv"


# ---------------------------------------------------------------------------
# Invariance: predictions on the fit data match whether or not we scale
# ---------------------------------------------------------------------------

def test_scaled_fitnlm_matches_unscaled_on_demo(analyzed_demo):
    """Scaled and unscaled fits describe the same response surface, so their
    predictions at training points must agree. Coefficients will differ
    (they live in different parameterizations), but y_pred must not.
    Tolerance: 1e-6 — linear regression is numerically well-conditioned on
    the demo data, so both paths converge to machine precision."""
    ivars = analyzed_demo["independent_vars"]
    dep_var = analyzed_demo["dependent_vars"][0]

    unscaled = run_analysis(
        dataframe=analyzed_demo["exp_df"],
        independent_vars=ivars,
        dependent_var=dep_var,
        model_type="Nonlinear LS (fitnlm)",
    )
    scaled = run_analysis(
        dataframe=analyzed_demo["exp_df"],
        independent_vars=ivars,
        dependent_var=dep_var,
        model_type="Nonlinear LS (fitnlm)",
        model_params={"scale_inputs": True},
    )

    y_un = np.asarray(unscaled.predict(analyzed_demo["exp_df"]))
    y_sc = np.asarray(scaled.predict(analyzed_demo["exp_df"]))
    np.testing.assert_allclose(y_un, y_sc, atol=1e-6)


def test_scaled_predict_accepts_raw_units(analyzed_demo):
    """The `predict(df)` contract: caller passes raw units and gets raw-unit
    predictions — the wrapper applies its internal scaler. Verify by feeding
    an out-of-fold row."""
    ivars = analyzed_demo["independent_vars"]
    dep_var = analyzed_demo["dependent_vars"][0]

    scaled = run_analysis(
        dataframe=analyzed_demo["exp_df"],
        independent_vars=ivars,
        dependent_var=dep_var,
        model_type="Nonlinear LS (fitnlm)",
        model_params={"scale_inputs": True},
    )
    # Typical dose in the raw units of the demo data
    test_row = pd.DataFrame({v: [float(analyzed_demo["exp_df"][v].mean())] for v in ivars})
    y_hat = scaled.predict(test_row)
    assert np.isfinite(y_hat).all()


# ---------------------------------------------------------------------------
# The main win: condition number drops by ~30 orders of magnitude on IOR1
# ---------------------------------------------------------------------------

def _design_matrix_cond(df, ivars, scale):
    """Compute the condition number of the polynomial design matrix used
    by OLS (same terms as the fitnlm polynomial), optionally with scaled
    inputs. Uses 2-norm condition number."""
    from sklearn.preprocessing import StandardScaler

    X_raw = df[ivars].values.astype(float)
    if scale:
        X_use = StandardScaler().fit_transform(X_raw)
        df_use = pd.DataFrame(X_use, columns=ivars)
    else:
        df_use = df[ivars].copy()
    _add_polynomial_terms(df_use, ivars)
    # statsmodels adds an intercept column; emulate that
    design = np.column_stack([np.ones(len(df_use)), df_use.values.astype(float)])
    return float(np.linalg.cond(design))


def test_scaling_reduces_condition_number_on_paper_ior1(capsys):
    """This is the headline result for Step 1. On the IOR1 paper data (raw
    concentrations 10^1 to 10^6 ng/mL), the unscaled polynomial design
    matrix has condition number ~10^38 — effectively singular to double
    precision. Standardizing inputs drops it by many orders of magnitude."""
    df_raw = pd.read_csv(IOR1_CSV)
    df = interpolate_to_n_rows(df_raw, n_target=15)
    (data_df, _, ivars, _, _, _, _, _, _) = analyze_csv(df)

    cond_raw = _design_matrix_cond(data_df, ivars, scale=False)
    cond_scaled = _design_matrix_cond(data_df, ivars, scale=True)

    print(f"\nIOR1 (n=15) design matrix condition number:")
    print(f"  raw units        = {cond_raw:.3e}")
    print(f"  standardized     = {cond_scaled:.3e}")
    print(f"  reduction factor = {cond_raw / cond_scaled:.3e}")

    # Scaling does not fix rank deficiency — the IOR1 design still has
    # duplicate rows from the day 4-6 and 10-13 plasma plateaus, which
    # keeps sigma_min ~ 0 and the condition number astronomically high.
    # But the MAGNITUDE problem (coefficients stretched over 10 orders of
    # magnitude) goes away, and that's what a downstream fit actually cares
    # about numerically. Expect a reduction factor of at least 10^10.
    assert cond_raw > 1e10, f"Expected raw cond# > 1e10, got {cond_raw}"
    assert cond_raw / cond_scaled > 1e10, (
        f"Scaling should reduce condition number by at least 10 orders, "
        f"got factor {cond_raw / cond_scaled:.2e}"
    )
