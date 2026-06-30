# src/logic/null_surface.py
"""
Non-interaction ("null" / reference) response surfaces and difference-to-neutral
synergy, built from per-drug marginal Hill fits (see logic.hill_fit).

This is the foundation of the mechanistic surface engine (Phase 1). It borrows
ComboKR's surface parameterization — marginal Hill fits -> null reference
surface -> difference-to-neutral — applied to the user's own measured data.

Conventions
-----------
Each drug i has a marginal Hill fit with its own (E0_i, Einf_i, EC50_i, h_i),
in the tool's *viability* convention: V_i(0) = E0_i (~100), V_i(inf) = Einf_i.
Per-drug fraction affected:
    fa_i(d) = (E0_i - V_i(d)) / (E0_i - Einf_i)   in [0, 1]

Null reference surfaces (n-drug native):
    Bliss : fa_ref = 1 - prod_i (1 - fa_i(d_i))    (independent action)
    HSA   : fa_ref = max_i fa_i(d_i)               (highest single agent)
    Loewe : solve  sum_i d_i / D_i(fa) = 1  for fa (additivity isobole),
            where D_i(fa) is the dose of drug i alone giving fraction fa
            (the inverse Hill, hill_fit.HillFit.mono_dose_for_target).

The reference fraction is mapped back to the user's response units with a
shared baseline (mean E0, mean Einf across the fitted drugs):
    V_ref = E0_bar - fa_ref * (E0_bar - Einf_bar)

Difference-to-neutral (synergy) surface, in response units:
    delta = V_model - V_ref
For viability/inhibition (lower V = stronger effect), delta < 0 means the
combination is *more* effective than the null expects = synergy.
"""

from __future__ import annotations
from typing import Dict, Sequence, Optional
import numpy as np
from scipy.optimize import brentq


# ── baseline helpers ────────────────────────────────────────────────────────────

def shared_baseline(hill_fits: Dict[str, "object"]) -> tuple:
    """Consensus (E0, Einf) across the per-drug Hill fits (they share an assay)."""
    e0 = float(np.mean([hf.E0 for hf in hill_fits.values()]))
    einf = float(np.mean([hf.Einf for hf in hill_fits.values()]))
    return e0, einf


def _fa_marginal(hill_fit, dose: float) -> float:
    """Per-drug fraction affected at `dose`, on the drug's own [0,1] range."""
    span = hill_fit.E0 - hill_fit.Einf
    if span <= 1e-12:
        return 0.0
    fa = (hill_fit.E0 - hill_fit.predict(float(dose))) / span
    return float(np.clip(fa, 0.0, 1.0))


# ── null reference fraction (per dose-combination) ──────────────────────────────

def _bliss_fa(doses: Dict[str, float], hill_fits: Dict[str, "object"]) -> float:
    surv = 1.0
    for drug, hf in hill_fits.items():
        surv *= (1.0 - _fa_marginal(hf, doses.get(drug, 0.0)))
    return 1.0 - surv


def _hsa_fa(doses: Dict[str, float], hill_fits: Dict[str, "object"]) -> float:
    return max((_fa_marginal(hf, doses.get(drug, 0.0))
                for drug, hf in hill_fits.items()), default=0.0)


def _loewe_fa(doses: Dict[str, float], hill_fits: Dict[str, "object"],
              e0: float, einf: float) -> float:
    """Solve the Loewe isobole  sum_i d_i / D_i(fa) = 1  for fa in (0,1).

    D_i(fa) = monotherapy dose of drug i giving the shared-baseline viability
    v(fa) = e0 - fa*(e0 - einf). Drugs at zero dose contribute nothing. Returns
    NaN when the isobole has no solution in range (e.g. a drug can't reach the
    required effect alone — antagonism-like / undefined region)."""
    active = {d: dose for d, dose in doses.items() if dose > 1e-12 and d in hill_fits}
    if not active:
        return 0.0

    def iso(fa: float) -> float:
        v = e0 - fa * (e0 - einf)
        total = 0.0
        for drug, dose in active.items():
            Di = hill_fits[drug].mono_dose_for_target(v)
            if Di is None or Di <= 0:
                return np.inf  # this drug alone cannot reach fa -> isobole undefined
            total += dose / Di
        return total - 1.0

    lo, hi = 1e-6, 1.0 - 1e-6
    try:
        f_lo, f_hi = iso(lo), iso(hi)
        if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
            return np.nan
        return float(brentq(iso, lo, hi, maxiter=200, xtol=1e-12))
    except (ValueError, RuntimeError):
        return np.nan


_FA_FUNCS = {"bliss": _bliss_fa, "hsa": _hsa_fa}


# ── public API ──────────────────────────────────────────────────────────────────

def reference_surface(method: str,
                      dose_rows: Sequence[Dict[str, float]],
                      hill_fits: Dict[str, "object"]) -> np.ndarray:
    """Null reference response (viability units) for each dose-combination row.

    method    : 'bliss' | 'hsa' | 'loewe'
    dose_rows : list of {drug_name: dose} dicts (a grid or the data points)
    hill_fits : {drug_name: HillFit} marginal fits
    """
    method = method.lower()
    e0, einf = shared_baseline(hill_fits)
    out = np.empty(len(dose_rows), dtype=float)
    for k, doses in enumerate(dose_rows):
        if method == "loewe":
            fa = _loewe_fa(doses, hill_fits, e0, einf)
        elif method in _FA_FUNCS:
            fa = _FA_FUNCS[method](doses, hill_fits)
        else:
            raise ValueError(f"Unknown null-surface method: {method!r} "
                             f"(use 'bliss', 'hsa', or 'loewe')")
        out[k] = e0 - fa * (e0 - einf)
    return out


def difference_to_neutral(model,
                          data_frame,
                          hill_fits: Dict[str, "object"],
                          method: str = "bliss") -> dict:
    """Difference-to-neutral synergy surface for a fitted model over data rows.

    delta = V_model - V_ref. For viability, delta < 0 = synergy (more effect
    than the null), delta > 0 = antagonism.

    Returns {'reference', 'model', 'delta'} as aligned 1-D arrays.
    """
    drugs = list(hill_fits.keys())
    dose_rows = [
        {d: float(data_frame[d].iloc[i]) for d in drugs}
        for i in range(len(data_frame))
    ]
    ref = reference_surface(method, dose_rows, hill_fits)
    model_v = np.asarray(model.predict(data_frame), dtype=float)
    return {"reference": ref, "model": model_v, "delta": model_v - ref}
