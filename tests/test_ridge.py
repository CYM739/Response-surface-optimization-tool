"""
Step 2 tests: RidgeWrapper (L2-regularized polynomial regression).

Ridge is the answer to the rank-deficiency problem that scaling (Step 1)
couldn't fix — the plateau rows in IOR1 leave the design matrix effectively
rank ~10 even with 15 interpolated rows and standardized inputs, and OLS
picks an arbitrary point on a flat SSR manifold. Ridge penalizes the
coefficient L2 norm, which pins down a unique minimum and suppresses the
wild edge-of-box extrapolation we saw in Step 1 (predicted tumor size
of -1444).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from logic.data_processing import analyze_csv, run_analysis
from logic.interpolation import interpolate_to_n_rows
from logic.models import RidgeWrapper


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
IOR1_CSV = _PROJECT_ROOT.parent / "Papers" / "ior1_dataset.csv"


# ---------------------------------------------------------------------------
# Interface sanity
# ---------------------------------------------------------------------------

def test_ridge_wrapper_has_required_interface(analyzed_demo):
    dep_var = analyzed_demo["dependent_vars"][0]
    ridge = run_analysis(
        dataframe=analyzed_demo["exp_df"],
        independent_vars=analyzed_demo["independent_vars"],
        dependent_var=dep_var,
        model_type="Ridge Regression",
    )
    assert isinstance(ridge, RidgeWrapper)
    assert ridge.independent_vars == analyzed_demo["independent_vars"]
    assert ridge.formula is not None
    assert ridge.alpha == 1e-3  # default
    assert ridge.r2_score is not None
    # Duck-typed interface shared with every other wrapper:
    assert hasattr(ridge, "predict")
    assert hasattr(ridge, "get_summary")


# ---------------------------------------------------------------------------
# alpha behavior — the core Ridge guarantee
# ---------------------------------------------------------------------------

def test_ridge_alpha_increases_shrinks_coefficient_norm(analyzed_demo):
    """Ridge's L2 penalty must shrink coefficients toward zero as alpha
    grows. This is the defining property of Ridge regression, and what
    makes it behave well on rank-deficient data."""
    dep_var = analyzed_demo["dependent_vars"][0]

    norms = []
    for alpha in [1e-6, 1e-3, 1.0, 100.0]:
        w = run_analysis(
            dataframe=analyzed_demo["exp_df"],
            independent_vars=analyzed_demo["independent_vars"],
            dependent_var=dep_var,
            model_type="Ridge Regression",
            model_params={"alpha": alpha, "scale_inputs": True},
        )
        # Skip the intercept — it's not penalized by Ridge
        norms.append(float(np.linalg.norm(w.params[1:])))

    # Monotone shrinkage: each successive alpha must yield a norm no larger
    # than the previous (with a tolerance for numerical noise at tiny alpha).
    for i in range(len(norms) - 1):
        assert norms[i + 1] <= norms[i] * 1.01 + 1e-9, (
            f"Ridge coef norm must shrink with alpha. "
            f"alpha indices {i}->{i+1}: {norms[i]:.4g} -> {norms[i+1]:.4g}"
        )
    # And the largest alpha should shrink substantially relative to the smallest
    assert norms[-1] < norms[0] * 0.5, (
        f"alpha=100 should shrink coefs by 2x+ vs alpha=1e-6 "
        f"(got {norms[0]:.4g} -> {norms[-1]:.4g})"
    )


# ---------------------------------------------------------------------------
# Rank-deficient data: what scaling alone couldn't fix
# ---------------------------------------------------------------------------

def test_ridge_produces_bounded_predictions_on_ior1(capsys):
    """The motivation for Ridge on this project. Unregularized fits of the
    IOR1 polynomial extrapolate to predicted tumor sizes of -54 (OLS) and
    -1444 (scaled fitnlm) at the edge of the dose box. Ridge with modest
    regularization must stay in a physically meaningful range."""
    df_raw = pd.read_csv(IOR1_CSV)
    df = interpolate_to_n_rows(df_raw, n_target=15)
    (data_df, _, ivars, dep_vars, _, _, _, _, _) = analyze_csv(df)

    wrapper = run_analysis(
        dataframe=data_df,
        independent_vars=ivars,
        dependent_var=dep_vars[0],
        model_type="Ridge Regression",
        model_params={"alpha": 1.0, "scale_inputs": True},
    )

    # Predict across the entire observed range + 30% — the exact box Step 0
    # and Step 1 tested against.
    from itertools import product
    grid_preds = []
    for corner in product(*[
        (float(data_df[v].min()), float(data_df[v].max()) * 1.3)
        for v in ivars
    ]):
        row = pd.DataFrame([dict(zip(ivars, corner))])
        grid_preds.append(float(wrapper.predict(row)[0]))

    gmin, gmax = min(grid_preds), max(grid_preds)
    print(f"\nRidge(alpha=1.0) IOR1 predictions across 2^4=16 box corners:")
    print(f"  min = {gmin:.3f}   max = {gmax:.3f}   span = {gmax - gmin:.3f}")
    print(f"  observed tumor range: {data_df[dep_vars[0]].min():.2f} - {data_df[dep_vars[0]].max():.2f}")

    # Predictions must stay within a sane multiple of the training range.
    # Training spans ~1.2-2.0; allow factor of 5 for extrapolation at 1.3x.
    y_min = float(data_df[dep_vars[0]].min())
    y_max = float(data_df[dep_vars[0]].max())
    training_span = y_max - y_min
    assert gmin > y_min - 5 * training_span, (
        f"Ridge extrapolated to pathologically low value: {gmin}"
    )
    assert gmax < y_max + 5 * training_span, (
        f"Ridge extrapolated to pathologically high value: {gmax}"
    )


def test_ridge_training_r2_spectrum_on_ior1():
    """Ridge's alpha selects a training-fit vs stability trade-off. Check
    both ends of the spectrum: default alpha=1e-3 recovers most of the
    training signal (R^2 > 0.9), while alpha=1.0 sacrifices R^2 in
    exchange for the bounded-extrapolation behavior the previous test
    verified (R^2 lands around 0.5-0.7, which is the correct ballpark —
    if the regularization is too weak it won't fix extrapolation, too
    strong it kills the signal)."""
    df_raw = pd.read_csv(IOR1_CSV)
    df = interpolate_to_n_rows(df_raw, n_target=15)
    (data_df, _, ivars, dep_vars, _, _, _, _, _) = analyze_csv(df)

    w_weak = run_analysis(
        dataframe=data_df, independent_vars=ivars, dependent_var=dep_vars[0],
        model_type="Ridge Regression",
        model_params={"alpha": 1e-3, "scale_inputs": True},
    )
    w_strong = run_analysis(
        dataframe=data_df, independent_vars=ivars, dependent_var=dep_vars[0],
        model_type="Ridge Regression",
        model_params={"alpha": 1.0, "scale_inputs": True},
    )
    assert w_weak.r2_score > 0.9, (
        f"Ridge(alpha=1e-3) should nearly recover unregularized fit, "
        f"got R^2={w_weak.r2_score:.3f}"
    )
    assert 0.3 < w_strong.r2_score < 0.85, (
        f"Ridge(alpha=1.0) should lose training fit but keep structure, "
        f"got R^2={w_strong.r2_score:.3f}"
    )
    assert w_weak.r2_score > w_strong.r2_score


# ---------------------------------------------------------------------------
# Predict contract — caller works in raw units
# ---------------------------------------------------------------------------

def test_ridge_predict_handles_single_row_dataframe(analyzed_demo):
    """objective_function in optimization.py passes a 1-row DataFrame —
    this codepath must not break."""
    dep_var = analyzed_demo["dependent_vars"][0]
    ivars = analyzed_demo["independent_vars"]
    wrapper = run_analysis(
        dataframe=analyzed_demo["exp_df"],
        independent_vars=ivars,
        dependent_var=dep_var,
        model_type="Ridge Regression",
    )
    single = pd.DataFrame({v: [float(analyzed_demo["exp_df"][v].mean())] for v in ivars})
    y = wrapper.predict(single)
    assert y.shape == (1,)
    assert np.isfinite(y).all()
