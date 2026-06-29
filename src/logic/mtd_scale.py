"""
Drug concentration <-> fraction-of-reference conversion.

Response-surface fitting is done in concentration space (the measured
biological quantity), but clinical and pharmacological reporting is almost
always in dose / fraction-of-MTD terms. When you optimize the PRS to pick
an optimal concentration vector, you still need to translate it back to
something comparable to literature values like 'Adriamycin at 1.0x MTD'.

This utility holds a per-drug reference concentration (e.g. the observed
plasma level at 1x MTD steady state, or any other anchor the user picks)
and does linear conversion in both directions. It is deliberately small
and does *not* model pharmacokinetics — dose-to-concentration for a real
drug depends on Vd, half-life, dosing schedule, etc. Users who need that
level of fidelity should plug in a proper PK model and feed its output
concentrations into this scale.

Typical use:
    ref = MTDScale({"Adriamycin": 112.0, "Herceptin": 837000.0, ...})
    df = ref.summarize({"Adriamycin": 123.0, "Herceptin": 784940.0, ...})
    # df['Fraction of reference'] is the comparable to literature values.
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd


class MTDScale:
    """Per-drug linear map between measured concentration and a unit-less
    fraction of a user-chosen reference concentration.

    Parameters
    ----------
    reference_concentrations : mapping
        {drug_name: concentration_at_1x_reference}. Keys are matched by
        exact string equality to column names in downstream DataFrames.
    """

    def __init__(self, reference_concentrations: Mapping[str, float]):
        if not reference_concentrations:
            raise ValueError("Need at least one drug reference concentration.")
        for drug, ref in reference_concentrations.items():
            if not isinstance(drug, str) or not drug:
                raise ValueError(f"Drug names must be non-empty strings (got {drug!r})")
            if not (ref > 0) or ref != ref:  # also catches NaN
                raise ValueError(
                    f"Reference concentration for '{drug}' must be > 0 (got {ref})"
                )
        self.reference = dict(reference_concentrations)

    @property
    def drugs(self) -> list[str]:
        return list(self.reference)

    def to_fraction(self, drug: str, concentration: float) -> float:
        if drug not in self.reference:
            raise KeyError(f"No reference concentration stored for '{drug}'.")
        return float(concentration) / float(self.reference[drug])

    def to_concentration(self, drug: str, fraction: float) -> float:
        if drug not in self.reference:
            raise KeyError(f"No reference concentration stored for '{drug}'.")
        return float(fraction) * float(self.reference[drug])

    def summarize(
        self,
        drug_concentrations: Mapping[str, float],
    ) -> pd.DataFrame:
        """Return a DataFrame with Drug / Concentration / Reference /
        Fraction columns. Missing drugs in the scale raise KeyError."""
        rows = []
        for drug, conc in drug_concentrations.items():
            rows.append({
                "Drug": drug,
                "Concentration": float(conc),
                "Reference (1.0x)": float(self.reference[drug]),
                "Fraction of reference": self.to_fraction(drug, conc),
            })
        return pd.DataFrame(rows)

    def compare(
        self,
        drug_concentrations: Mapping[str, float],
        literature_fractions: Mapping[str, float],
    ) -> pd.DataFrame:
        """Side-by-side comparison: our optimum (converted to fraction) vs
        a reference/literature table of fractions. Useful for reproducibility
        reports — 'did our PRS optimum land in the same neighborhood as the
        published regimen?'."""
        rows = []
        for drug, conc in drug_concentrations.items():
            ours = self.to_fraction(drug, conc)
            theirs = literature_fractions.get(drug)
            rows.append({
                "Drug": drug,
                "Our optimum (fraction)": ours,
                "Literature (fraction)": theirs,
                "Delta": (ours - theirs) if theirs is not None else None,
            })
        return pd.DataFrame(rows)
