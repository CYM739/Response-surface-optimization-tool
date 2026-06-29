"""
Tier 0: direct tests against the logic layer.

No Streamlit runtime, no browser. An agent edits logic code and gets a
pass/fail signal in milliseconds. Exercises the same call chain that the
views use: analyze_csv -> run_analysis -> objective_function / run_optimization.
"""
import numpy as np
import pandas as pd
import pytest

from logic.data_processing import analyze_csv, run_analysis
from logic.models import OLSWrapper
from logic.optimization import objective_function, run_optimization


# ---------------------------------------------------------------------------
# analyze_csv
# ---------------------------------------------------------------------------

def test_analyze_csv_classifies_demo_variables(analyzed_demo):
    """DrugA/B/C should be independent; Cell_Viability should be dependent
    (prefix 'Cell_' is on the dependent-variable list in data_processing.py)."""
    assert analyzed_demo["independent_vars"] == ["DrugA", "DrugB", "DrugC"]
    assert analyzed_demo["dependent_vars"] == ["Cell_Viability"]


def test_analyze_csv_records_variable_stats(analyzed_demo):
    """variable_stats[var] = (min, second_min, max). Optimizer bounds use
    second_min as the lower default, so this tuple shape matters."""
    for var in analyzed_demo["independent_vars"]:
        stats = analyzed_demo["variable_stats"][var]
        assert len(stats) == 3
        min_v, second_min_v, max_v = stats
        assert min_v <= second_min_v <= max_v


def test_analyze_csv_no_binary_vars_in_demo(analyzed_demo):
    """The demo doses are continuous — binary detection should be empty."""
    assert analyzed_demo["detected_binary_vars"] == []


# ---------------------------------------------------------------------------
# OLS fit
# ---------------------------------------------------------------------------

def test_ols_wrapper_predicts_training_rows(trained_ols, analyzed_demo):
    """R^2 on the training set should be reasonable — if this drops below 0.7
    the demo data or the model builder has regressed."""
    wrapper = trained_ols["wrapper"]
    dep_var = trained_ols["dep_var"]
    df = analyzed_demo["exp_df"]

    assert isinstance(wrapper, OLSWrapper)
    assert wrapper.independent_vars == analyzed_demo["independent_vars"]

    preds = wrapper.predict(df)
    actual = df[dep_var].values
    ss_res = float(np.sum((actual - preds) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    assert r2 > 0.7, f"Unexpectedly low R^2 on demo data: {r2:.3f}"


def test_ols_summary_is_non_empty(trained_ols):
    summary = trained_ols["wrapper"].get_summary()
    assert "OLS Regression Results" in summary


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def test_objective_function_matches_wrapper_predict(trained_ols, analyzed_demo):
    """objective_function must agree with wrapper.predict for the same x vector
    — the optimizer's math depends on this equivalence."""
    wrapper = trained_ols["wrapper"]
    ivars = analyzed_demo["independent_vars"]
    x = np.array([1.0, 0.5, 2.0])

    obj = objective_function(x, wrapper, ivars)
    pred = wrapper.predict(pd.DataFrame([x], columns=ivars)).iloc[0]
    assert obj == pytest.approx(pred, rel=1e-6)


def test_slsqp_finds_low_viability_point(trained_ols, analyzed_demo):
    """Minimizing Cell_Viability with SLSQP over the dose box should land on a
    point where predicted viability is notably below the untreated baseline
    (~100 at dose (0,0,0))."""
    wrapper = trained_ols["wrapper"]
    ivars = analyzed_demo["independent_vars"]

    bounds = [(0.0, 50.0), (0.0, 20.0), (0.0, 20.0)]
    start = [b[0] for b in bounds]

    result = run_optimization(
        fun=lambda x: objective_function(x, wrapper, ivars),
        bounds=bounds,
        start_points=start,
        constraints=[],
        algorithm="SLSQP (Local)",
        algo_params={},
    )

    assert result.success, f"SLSQP failed: {result.message}"
    baseline = float(
        wrapper.predict(pd.DataFrame([[0, 0, 0]], columns=ivars)).iloc[0]
    )
    assert result.fun < baseline, (
        f"Optimizer did not improve on untreated baseline: "
        f"opt={result.fun:.2f} baseline={baseline:.2f}"
    )
