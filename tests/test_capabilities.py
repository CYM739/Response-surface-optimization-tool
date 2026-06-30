"""Model capability flags and the generalized (non-OLS) diagnostics path."""
import numpy as np
import pandas as pd
import pytest

from logic.models import (
    OLSWrapper, NonlinearLSWrapper, RidgeWrapper,
    SVRWrapper, RandomForestWrapper,
)
from logic.diagnostics import (
    build_design_matrix, perform_heteroscedasticity_test,
    perform_normality_test, perform_autocorrelation_test,
)


def test_capability_flags():
    assert OLSWrapper.IS_PARAMETRIC_POLYNOMIAL
    assert OLSWrapper.SUPPORTS_GRADIENT_OPT
    assert OLSWrapper.SUPPORTS_OLS_INFERENCE

    for W in (NonlinearLSWrapper, RidgeWrapper):
        assert W.IS_PARAMETRIC_POLYNOMIAL
        assert W.SUPPORTS_GRADIENT_OPT
        assert not W.SUPPORTS_OLS_INFERENCE          # diagnostics yes, OLS-inference no

    assert not SVRWrapper.IS_PARAMETRIC_POLYNOMIAL   # no diagnostics
    assert SVRWrapper.SUPPORTS_GRADIENT_OPT          # but smooth -> optimizer ok

    assert not RandomForestWrapper.IS_PARAMETRIC_POLYNOMIAL
    assert not RandomForestWrapper.SUPPORTS_GRADIENT_OPT   # non-smooth -> AI optimizer only


def _toy(n=40, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.uniform(0, 10, n)
    B = rng.uniform(0, 10, n)
    y = 100 - 3 * A - 2 * B + 0.1 * A * B + rng.normal(0, 1.0, n)
    return pd.DataFrame({"A": A, "B": B, "Cell_Viability": y})


def test_build_design_matrix_has_poly_terms_and_constant():
    df = _toy()
    X = build_design_matrix(df, ["A", "B"])
    cols = set(X.columns)
    assert "const" in cols
    assert {"A", "B"} <= cols
    assert any(c.endswith("_sq") for c in cols)       # squared terms
    assert any("*" in c for c in cols)                # interaction term
    assert X.shape[0] == len(df)


@pytest.mark.parametrize("Wrap", [NonlinearLSWrapper, RidgeWrapper])
def test_generic_diagnostics_path_for_non_ols(Wrap):
    """fitnlm/Ridge get residual-based assumption tests via generic residuals + exog."""
    df = _toy()
    iv = ["A", "B"]
    w = Wrap(iv)
    w.fit(df, "Cell_Viability")

    resid = df["Cell_Viability"].to_numpy(float) - np.asarray(w.predict(df), float)
    exog = build_design_matrix(df, iv)

    bp_p = perform_heteroscedasticity_test(resid, exog=exog)
    assert 0.0 <= bp_p <= 1.0

    _, norm_p, _ = perform_normality_test(resid)
    assert 0.0 <= norm_p <= 1.0

    dw = perform_autocorrelation_test(resid)
    assert 0.0 <= dw <= 4.0
