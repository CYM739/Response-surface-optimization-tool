# src/logic/hill_fit.py
"""
Marginal Hill curve fitting and Dose Reduction Index (DRI) computation.

Pure functions — no Streamlit imports.

When a user's dataset includes monotherapy rows (all other drugs at 0),
this module fits a 4-parameter Hill equation per drug:

    V(d) = Einf + (E0 - Einf) / (1 + (d / EC50) ^ h)

where:
    E0   = viability at zero dose (baseline, ~100%)
    Einf = asymptotic viability at infinite dose (the kill floor)
    EC50 = dose producing 50% of the maximum effect
    h    = Hill slope (steepness)

After optimization finds (d1*, d2*, V*), DRI answers:
    "How much of drug X alone would be needed to reach V*?"
    DRI = d_mono_equivalent / d_combo_optimal

DRI > 1  → combination uses less drug than monotherapy (synergy benefit)
DRI < 1  → combination needs MORE drug than monotherapy (antagonism or inefficiency)
DRI = inf → drug alone cannot reach V* (below Einf floor) — strongest possible signal
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class HillFit:
    """Result of fitting a 4-parameter Hill curve to monotherapy edge data."""
    drug: str
    dep_var: str
    EC50: float
    h: float
    E0: float          # baseline viability (d=0)
    Einf: float        # asymptotic viability (d→∞)
    r2: float
    n_points: int
    success: bool
    error: str = ""

    def predict(self, dose: float) -> float:
        """Predict viability at a given single-drug dose."""
        if dose <= 0:
            return float(self.E0)
        T = float(np.clip((dose / self.EC50) ** self.h, 0, 1e8))
        return float(self.Einf + (self.E0 - self.Einf) / (1.0 + T))

    def mono_dose_for_target(self, target_v: float) -> Optional[float]:
        """
        Analytically invert the Hill curve:
        return the monotherapy dose needed to achieve target_v viability.
        Returns None when target_v <= Einf (impossible for this drug alone).
        """
        if target_v <= self.Einf:
            return None
        ratio = (self.E0 - self.Einf) / (target_v - self.Einf) - 1.0
        if ratio <= 0:
            return None
        return float(self.EC50 * (ratio ** (1.0 / self.h)))


# ── Fitting ────────────────────────────────────────────────────────────────────

def _hill_4param(d, EC50, h, Einf, E0_fixed):
    """4-parameter Hill equation with E0 fixed externally."""
    T = np.clip((d / EC50) ** h, 0, 1e8)
    return Einf + (E0_fixed - Einf) / (1.0 + T)


def fit_hill_curve(doses: np.ndarray, viabilities: np.ndarray,
                   drug: str = "", dep_var: str = "") -> HillFit:
    """
    Fit a 4-parameter Hill curve (E0 fixed to observed baseline).
    Requires >= 3 strictly positive dose points.

    doses       : 1-D array of dose values (should include 0 for E0 estimate)
    viabilities : 1-D array of viability values (same length, any scale)
    """
    doses = np.asarray(doses, dtype=float)
    viabilities = np.asarray(viabilities, dtype=float)

    # Remove NaN/inf
    valid = np.isfinite(doses) & np.isfinite(viabilities)
    doses = doses[valid]
    viabilities = viabilities[valid]

    # Estimate E0 from zero-dose rows; fallback to max viability
    zero_mask = doses < 1e-9
    E0_est = float(viabilities[zero_mask].mean()) if zero_mask.any() else float(viabilities.max())

    # Only use positive-dose rows for fitting
    pos = doses > 1e-9
    d_fit = doses[pos]
    v_fit = viabilities[pos]

    if len(d_fit) < 3:
        return HillFit(drug=drug, dep_var=dep_var, EC50=np.nan, h=np.nan,
                       E0=E0_est, Einf=np.nan, r2=np.nan,
                       n_points=int(pos.sum()), success=False,
                       error="Need >= 3 positive-dose points")

    # Scale: detect if viability is 0-1 or 0-100
    scale = 100.0 if E0_est > 1.5 else 1.0

    # Bounds and initial guess
    EC50_guess = float(np.median(d_fit))
    Einf_lo    = 0.0 if scale > 1.5 else 0.0
    Einf_hi    = float(E0_est * 0.99)   # Einf must be below E0

    try:
        popt, pcov = curve_fit(
            lambda d, EC50, h, Einf: _hill_4param(d, EC50, h, Einf, E0_est),
            d_fit, v_fit,
            p0=[EC50_guess, 1.5, E0_est * 0.2],
            bounds=([1e-12, 0.1, Einf_lo], [1e12, 10.0, Einf_hi]),
            maxfev=5000,
        )
        EC50_fit, h_fit, Einf_fit = popt

        # R²
        v_pred = _hill_4param(d_fit, EC50_fit, h_fit, Einf_fit, E0_est)
        ss_res = np.sum((v_fit - v_pred) ** 2)
        ss_tot = np.sum((v_fit - v_fit.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan

        return HillFit(
            drug=drug, dep_var=dep_var,
            EC50=float(EC50_fit), h=float(h_fit),
            E0=float(E0_est), Einf=float(Einf_fit),
            r2=r2, n_points=int(pos.sum()), success=True,
        )

    except Exception as e:
        return HillFit(drug=drug, dep_var=dep_var, EC50=np.nan, h=np.nan,
                       E0=E0_est, Einf=np.nan, r2=np.nan,
                       n_points=int(pos.sum()), success=False,
                       error=str(e))


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_and_fit_hills(df: pd.DataFrame,
                         independent_vars: list,
                         dependent_vars: list,
                         min_points: int = 3) -> dict:
    """
    Detect monotherapy edge rows in df and fit Hill curves.

    Returns a nested dict:
        { dep_var: { drug_name: HillFit } }

    Only populated when sufficient edge data exists.
    Completely silent on failure — never raises.
    """
    results = {}

    if len(independent_vars) < 2:
        return results  # No combination data — monotherapy edge concept doesn't apply

    for dep_var in dependent_vars:
        if dep_var not in df.columns:
            continue

        fits = {}
        for drug in independent_vars:
            try:
                # Edge rows: this drug > 0, all other drugs == 0
                other_drugs = [v for v in independent_vars if v != drug]
                mask = pd.to_numeric(df[drug], errors="coerce") > 1e-9
                for other in other_drugs:
                    mask &= pd.to_numeric(df[other], errors="coerce").abs() < 1e-9

                edge_df = df[mask].copy()
                if len(edge_df) < min_points:
                    continue

                # Also include zero-dose row for E0 estimation
                zero_mask = pd.Series([True] * len(df), index=df.index)
                for v in independent_vars:
                    zero_mask &= pd.to_numeric(df[v], errors="coerce").abs() < 1e-9
                zero_df = df[zero_mask].copy()

                combined = pd.concat([zero_df, edge_df])
                doses = pd.to_numeric(combined[drug], errors="coerce").values
                viabilities = pd.to_numeric(combined[dep_var], errors="coerce").values

                hf = fit_hill_curve(doses, viabilities, drug=drug, dep_var=dep_var)
                if hf.success:
                    fits[drug] = hf

            except Exception:
                continue  # Never break the main app

        if fits:
            results[dep_var] = fits

    return results


# ── DRI computation ────────────────────────────────────────────────────────────

def compute_dri(hill_fit: HillFit, d_combo: float, v_target: float) -> dict:
    """
    Compute Dose Reduction Index for one drug.

    d_combo  : optimal combination dose for this drug
    v_target : viability at the optimal combination point

    Returns dict with:
        mono_dose  : equivalent monotherapy dose (None if not achievable)
        dri        : mono_dose / d_combo  (None if mono_dose is None or d_combo==0)
        achievable : bool — whether monotherapy can reach v_target
    """
    mono = hill_fit.mono_dose_for_target(v_target)
    achievable = mono is not None

    if not achievable:
        return {"mono_dose": None, "dri": None, "achievable": False}

    if d_combo <= 1e-12:
        return {"mono_dose": mono, "dri": None, "achievable": True}

    return {
        "mono_dose": mono,
        "dri":       mono / d_combo,
        "achievable": True,
    }
