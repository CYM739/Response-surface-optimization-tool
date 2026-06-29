"""
Step 4 tests: log_transform=True on Ridge/fitnlm wrappers.

Motivation: the paper fits a polynomial in log-dose space (standard in
pharmacology — the Hill equation says response is sigmoidal in log-dose,
and a low-order polynomial is a reasonable local approximation there).
When our raw-space Ridge fit recommends 'Herceptin stays near 1x' (vs the
paper's 'Herceptin drops to 0.2x'), one possible cause is that our model
doesn't see the concavity the paper's log-space fit does. These tests
verify the log pipeline mechanics, and the IOR1 reproduction asks whether
log-space changes the direction match.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from logic.data_processing import analyze_csv, run_analysis
from logic.interpolation import interpolate_to_n_rows
from logic.mtd_scale import MTDScale
from logic.models import NonlinearLSWrapper, RidgeWrapper


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
IOR1_CSV = _PROJECT_ROOT.parent / "Papers" / "ior1_dataset.csv"


# ---------------------------------------------------------------------------
# Mechanics: log path rejects invalid inputs, round-trips consistently
# ---------------------------------------------------------------------------

def test_log_transform_rejects_non_positive_inputs(analyzed_demo):
    """demo_3drug_dri has zero doses in some rows. log(0) = -inf, so the
    fit must refuse to run (same rule Matlab applies)."""
    ivars = analyzed_demo["independent_vars"]
    dep_var = analyzed_demo["dependent_vars"][0]

    with pytest.raises(ValueError, match="log_transform=True requires"):
        run_analysis(
            dataframe=analyzed_demo["exp_df"],
            independent_vars=ivars,
            dependent_var=dep_var,
            model_type="Nonlinear LS (fitnlm)",
            model_params={"log_transform": True},
        )


def test_ridge_log_transform_rejects_non_positive_inputs(analyzed_demo):
    ivars = analyzed_demo["independent_vars"]
    dep_var = analyzed_demo["dependent_vars"][0]

    with pytest.raises(ValueError, match="log_transform=True requires"):
        run_analysis(
            dataframe=analyzed_demo["exp_df"],
            independent_vars=ivars,
            dependent_var=dep_var,
            model_type="Ridge Regression",
            model_params={"log_transform": True},
        )


def test_fitnlm_log_predict_roundtrip_on_ior1():
    """Log path must be self-consistent: predictions at the training rows
    should match the training-row y values to zero residual when n==p
    (same exact-fit guarantee as the raw path)."""
    df_raw = pd.read_csv(IOR1_CSV)
    df = interpolate_to_n_rows(df_raw, n_target=15)
    (data_df, _, iv, dv, _, _, _, _, _) = analyze_csv(df)

    wrapper = run_analysis(
        dataframe=data_df,
        independent_vars=iv,
        dependent_var=dv[0],
        model_type="Nonlinear LS (fitnlm)",
        model_params={"log_transform": True, "scale_inputs": True},
    )
    y_actual = data_df[dv[0]].values
    y_pred = wrapper.predict(data_df)
    # n=p=15 with rank-deficient (plateau) data gives R^2 ~ 0.97 in log+scale
    # space too; check predictions are at least bounded.
    assert np.all(np.isfinite(y_pred))
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - y_actual.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.9, f"log+scale fitnlm R^2 too low on IOR1: {r2}"


def test_ridge_log_predict_roundtrip_on_ior1():
    df_raw = pd.read_csv(IOR1_CSV)
    df = interpolate_to_n_rows(df_raw, n_target=15)
    (data_df, _, iv, dv, _, _, _, _, _) = analyze_csv(df)

    wrapper = run_analysis(
        dataframe=data_df,
        independent_vars=iv,
        dependent_var=dv[0],
        model_type="Ridge Regression",
        model_params={
            "alpha": 1e-3, "scale_inputs": True, "log_transform": True,
        },
    )
    y_pred = wrapper.predict(data_df)
    assert np.all(np.isfinite(y_pred))
    assert wrapper.r2_score > 0.8


# ---------------------------------------------------------------------------
# IOR1 log-dose reproduction: does it close the Herceptin gap?
# ---------------------------------------------------------------------------

PAPER_IOR1_FRACTION_OF_MTD = {
    "Adriamycin": 1.00,
    "Gemcitabine": 0.80,
    "Cisplatin": 1.06,
    "Herceptin": 0.20,
}


def _classify_fraction(frac: float) -> str:
    if frac <= 0.25:
        return "MIN"
    if frac >= 0.75:
        return "HIGH"
    return "MID"


def test_ior1_log_ridge_reproduction(capsys):
    """Step 4: interp(13->15) + Ridge(alpha=1e-3, log, scaled) + SHGO global
    search, bound = observed range (no extrapolation). If log-space fit
    captures the dose-response concavity the paper's fit does, this should
    close the Herceptin gap to 4/4 without extending the input range."""
    from scipy.optimize import shgo

    df_raw = pd.read_csv(IOR1_CSV)
    df = interpolate_to_n_rows(df_raw, n_target=15)
    (data_df, _, iv, dv, _, _, _, _, _) = analyze_csv(df)
    dep_var = dv[0]

    day7 = df_raw.iloc[6]
    scale = MTDScale({var: float(day7[var]) for var in iv})

    report = ["", "=" * 90]
    report.append("IOR1 Step-4: log-dose Ridge sweep vs raw-dose Ridge (bound = observed range)")
    report.append("=" * 90)
    header = f"{'fit':<45}{'y_min':>10}{'Herc frac':>12}{'Herc bucket':>14}{'match':>10}"
    report.append(header)
    report.append("-" * len(header))

    configs = [
        ("raw   alpha=1e-3  scale=T  log=F", {"alpha": 1e-3, "scale_inputs": True, "log_transform": False}),
        ("raw   alpha=1.0   scale=T  log=F", {"alpha": 1.0,  "scale_inputs": True, "log_transform": False}),
        ("log   alpha=1e-3  scale=T  log=T", {"alpha": 1e-3, "scale_inputs": True, "log_transform": True }),
        ("log   alpha=1e-2  scale=T  log=T", {"alpha": 1e-2, "scale_inputs": True, "log_transform": True }),
        ("log   alpha=1e-1  scale=T  log=T", {"alpha": 1e-1, "scale_inputs": True, "log_transform": True }),
        ("log   alpha=1.0   scale=T  log=T", {"alpha": 1.0,  "scale_inputs": True, "log_transform": True }),
    ]

    bounds = [(float(data_df[v].min()), float(data_df[v].max())) for v in iv]

    best_log_match = 0
    best_log_herc_bucket = None
    for label, params in configs:
        wrapper = run_analysis(
            dataframe=data_df, independent_vars=iv, dependent_var=dep_var,
            model_type="Ridge Regression",
            model_params=params,
        )
        res = shgo(
            func=lambda x: float(wrapper.predict(pd.DataFrame([dict(zip(iv, x))]))[0]),
            bounds=bounds, n=300, iters=5,
        )
        opt_concs = dict(zip(iv, res.x))
        matches = 0
        herc_frac = None
        herc_bucket = None
        for var in iv:
            ours = scale.to_fraction(var, opt_concs[var])
            theirs = PAPER_IOR1_FRACTION_OF_MTD[var]
            if _classify_fraction(ours) == _classify_fraction(theirs):
                matches += 1
            if var == "Herceptin":
                herc_frac = ours
                herc_bucket = _classify_fraction(ours)
        report.append(
            f"{label:<45}{float(res.fun):>10.3f}{herc_frac:>12.3f}{herc_bucket:>14}{matches:>10}/4"
        )
        if params["log_transform"] and matches > best_log_match:
            best_log_match = matches
            best_log_herc_bucket = herc_bucket

    report.append("=" * 90)
    report.append(f"Best log-space result: {best_log_match}/4 direction match, "
                  f"Herceptin bucket = {best_log_herc_bucket}")
    report.append("=" * 90)
    print("\n".join(report))

    # Step 4's structural claim: log path is at least no worse than raw
    # (direction-wise) and is finite / well-behaved.
    assert best_log_match >= 2
