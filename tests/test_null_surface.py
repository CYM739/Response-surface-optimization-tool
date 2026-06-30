"""Tests for logic.null_surface — Bliss/HSA/Loewe reference surfaces and
difference-to-neutral, validated against closed-form expectations."""
import numpy as np
import pandas as pd
import pytest

from logic.hill_fit import HillFit
from logic.null_surface import reference_surface, difference_to_neutral


def _hf(drug, EC50, h, E0=100.0, Einf=0.0):
    return HillFit(drug=drug, dep_var="Cell_Viability", EC50=EC50, h=h,
                   E0=E0, Einf=Einf, r2=1.0, n_points=5, success=True)


def _grid(drugs, doses):
    """All combinations of the per-drug dose vectors -> list of dose dicts."""
    mesh = np.meshgrid(*doses, indexing="ij")
    flat = [m.ravel() for m in mesh]
    return [{d: float(flat[i][k]) for i, d in enumerate(drugs)}
            for k in range(flat[0].size)]


# ── monotherapy edges: every method reduces to the single-drug curve ────────────

def test_monotherapy_edges_equal_marginal_for_all_methods():
    fits = {"A": _hf("A", 10, 1.2), "B": _hf("B", 20, 0.9)}
    rows = [{"A": 5.0, "B": 0.0}, {"A": 0.0, "B": 7.0}, {"A": 0.0, "B": 0.0}]
    vA = fits["A"].predict(5.0)
    vB = fits["B"].predict(7.0)
    for method in ("bliss", "hsa", "loewe"):
        ref = reference_surface(method, rows, fits)
        assert ref[0] == pytest.approx(vA, abs=1e-6), method
        assert ref[1] == pytest.approx(vB, abs=1e-6), method
        assert ref[2] == pytest.approx(100.0, abs=1e-6), method  # no drug -> E0


# ── Bliss matches the closed form (E0=100, Einf=0 -> V_ref = prod V_i / 100^(n-1)) ─

def test_bliss_matches_closed_form_2drug():
    fits = {"A": _hf("A", 10, 1.5), "B": _hf("B", 25, 1.0)}
    drugs = list(fits)
    rows = _grid(drugs, [np.array([0, 2, 8, 30.0]), np.array([0, 3, 12, 40.0])])
    ref = reference_surface("bliss", rows, fits)
    expect = np.array([fits["A"].predict(r["A"]) * fits["B"].predict(r["B"]) / 100.0
                       for r in rows])
    assert np.allclose(ref, expect, atol=1e-6)


def test_hsa_matches_closed_form():
    fits = {"A": _hf("A", 10, 1.5), "B": _hf("B", 25, 1.0)}
    rows = _grid(list(fits), [np.array([0, 5, 20.0]), np.array([0, 5, 20.0])])
    ref = reference_surface("hsa", rows, fits)
    # HSA = strongest single agent = LOWEST viability of the marginals
    expect = np.array([min(fits["A"].predict(r["A"]), fits["B"].predict(r["B"]))
                       for r in rows])
    assert np.allclose(ref, expect, atol=1e-6)


# ── Loewe identity: two identical drugs at (d,d) == single drug at 2d ────────────

def test_loewe_identical_drugs_dose_additivity():
    fits = {"A": _hf("A", 15, 1.3), "B": _hf("B", 15, 1.3)}  # identical
    single = _hf("A", 15, 1.3)
    rows = [{"A": d, "B": d} for d in (1.0, 5.0, 15.0, 50.0)]
    ref = reference_surface("loewe", rows, fits)
    expect = np.array([single.predict(2 * d) for d in (1.0, 5.0, 15.0, 50.0)])
    assert np.allclose(ref, expect, atol=1e-3)


# ── n-drug (3) Bliss is native ──────────────────────────────────────────────────

def test_three_drug_bliss():
    fits = {"A": _hf("A", 10, 1.0), "B": _hf("B", 20, 1.0), "C": _hf("C", 30, 1.0)}
    rows = [{"A": 10.0, "B": 20.0, "C": 30.0}, {"A": 5.0, "B": 0.0, "C": 30.0}]
    ref = reference_surface("bliss", rows, fits)
    expect = np.array([
        fits["A"].predict(r["A"]) * fits["B"].predict(r["B"]) * fits["C"].predict(r["C"]) / 100.0**2
        for r in rows])
    assert np.allclose(ref, expect, atol=1e-6)


# ── difference-to-neutral sign convention ───────────────────────────────────────

class _StubModel:
    def __init__(self, fn):
        self._fn = fn
    def predict(self, df):
        return self._fn(df)


def test_difference_to_neutral_zero_for_additive_model():
    fits = {"A": _hf("A", 10, 1.5), "B": _hf("B", 25, 1.0)}
    df = pd.DataFrame({"A": [0, 5, 20, 5.0], "B": [0, 5, 5, 20.0]})
    # A model whose surface IS the Bliss reference -> delta ~ 0
    rows = [{"A": float(df["A"].iloc[i]), "B": float(df["B"].iloc[i])} for i in range(len(df))]
    bliss = reference_surface("bliss", rows, fits)
    model = _StubModel(lambda d: bliss)
    out = difference_to_neutral(model, df, fits, method="bliss")
    assert np.allclose(out["delta"], 0.0, atol=1e-9)


def test_difference_to_neutral_negative_for_synergy():
    fits = {"A": _hf("A", 10, 1.5), "B": _hf("B", 25, 1.0)}
    df = pd.DataFrame({"A": [5, 20.0], "B": [5, 10.0]})
    rows = [{"A": float(df["A"].iloc[i]), "B": float(df["B"].iloc[i])} for i in range(len(df))]
    bliss = reference_surface("bliss", rows, fits)
    # Model kills 10 viability units MORE than additive everywhere -> synergy
    model = _StubModel(lambda d: bliss - 10.0)
    out = difference_to_neutral(model, df, fits, method="bliss")
    assert np.all(out["delta"] < 0)
    assert np.allclose(out["delta"], -10.0, atol=1e-9)
