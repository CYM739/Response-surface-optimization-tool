"""MechanisticWrapper — MuSyC / BRAID 2-drug surfaces via the `synergy` package."""
import numpy as np
import pandas as pd
import pytest

from logic.data_processing import run_analysis
from logic.models import MechanisticWrapper


def _synergy_df(kappa=3.0, seed=0):
    """BRAID surface with kappa>1 (synergy) + monotherapy edges."""
    rng = np.random.default_rng(seed)
    E0, Einf, EC1, EC2, h1, h2 = 100.0, 0.0, 8.0, 15.0, 1.4, 1.1
    d1s = [0, 1, 2, 4, 8, 16, 32]
    d2s = [0, 2, 4, 8, 15, 30, 60]
    rows = []
    for a in d1s:
        for b in d2s:
            T1 = (a / EC1) ** h1 if a > 0 else 0.0
            T2 = (b / EC2) ** h2 if b > 0 else 0.0
            v = Einf + (E0 - Einf) / (1 + T1 + T2 + kappa * T1 * T2) + rng.normal(0, 1.0)
            rows.append({"DrugA": a, "DrugB": b, "Cell_Viability": float(np.clip(v, 0, 110))})
    return pd.DataFrame(rows)


def test_capability_flags():
    assert not MechanisticWrapper.IS_PARAMETRIC_POLYNOMIAL
    assert MechanisticWrapper.SUPPORTS_GRADIENT_OPT
    assert not MechanisticWrapper.SUPPORTS_OLS_INFERENCE


def test_requires_exactly_two_drugs():
    with pytest.raises(ValueError):
        MechanisticWrapper(["A", "B", "C"], kind="musyc")
    with pytest.raises(ValueError):
        MechanisticWrapper(["A"], kind="braid")


@pytest.mark.parametrize("model_type,kind", [
    ("MuSyC (mechanistic)", "musyc"),
    ("BRAID (2-drug)", "braid"),
])
def test_fits_bounded_surface_and_reads_synergy(model_type, kind):
    df = _synergy_df()
    w = run_analysis(df, ["DrugA", "DrugB"], "Cell_Viability", model_type)

    assert isinstance(w, MechanisticWrapper)
    assert w.r2_score > 0.9

    # Bounded by construction: a polynomial OLS on the same data extrapolates to
    # ~ -16 viability; the mechanistic surface should stay near the physical floor.
    pred = np.asarray(w.predict(df), float)
    assert pred.min() >= -2.0
    assert pred.max() <= 105.0

    # Synergy comes straight from the fitted parameters.
    if kind == "musyc":
        amean = 0.5 * (float(w.params["alpha12"]) + float(w.params["alpha21"]))
        assert amean > 1.0
    else:
        assert float(w.params["kappa"]) > 0.0
    assert "Synergistic" in w.get_summary()


def test_synergy_metrics_and_flag():
    df = _synergy_df()
    w = run_analysis(df, ["DrugA", "DrugB"], "Cell_Viability", "MuSyC (mechanistic)")
    assert getattr(w, "IS_MECHANISTIC", False)
    rows = w.synergy_metrics()
    assert all(set(r) == {"Parameter", "Value", "Baseline", "Verdict"} for r in rows)
    # potency synergy (alpha > 1) for this kappa>1 data
    assert any(r["Parameter"].startswith("alpha") and r["Verdict"] == "Synergistic" for r in rows)

    wb = run_analysis(df, ["DrugA", "DrugB"], "Cell_Viability", "BRAID (2-drug)")
    assert any(r["Parameter"].startswith("kappa") for r in wb.synergy_metrics())
