"""Unit tests for logic.mtd_scale.MTDScale."""
import pandas as pd
import pytest

from logic.mtd_scale import MTDScale


def test_round_trip_is_identity():
    scale = MTDScale({"A": 100.0, "B": 200.0})
    for drug, ref in [("A", 100.0), ("B", 200.0)]:
        for frac in [0.0, 0.5, 1.0, 1.33]:
            conc = scale.to_concentration(drug, frac)
            back = scale.to_fraction(drug, conc)
            assert back == pytest.approx(frac)


def test_unknown_drug_raises():
    scale = MTDScale({"A": 100.0})
    with pytest.raises(KeyError):
        scale.to_fraction("B", 50.0)
    with pytest.raises(KeyError):
        scale.to_concentration("B", 0.5)


def test_empty_reference_rejected():
    with pytest.raises(ValueError):
        MTDScale({})


def test_non_positive_reference_rejected():
    with pytest.raises(ValueError):
        MTDScale({"A": 0.0})
    with pytest.raises(ValueError):
        MTDScale({"A": -5.0})


def test_summarize_produces_expected_columns():
    scale = MTDScale({"A": 100.0, "B": 200.0})
    df = scale.summarize({"A": 120.0, "B": 100.0})
    assert list(df.columns) == [
        "Drug", "Concentration", "Reference (1.0x)", "Fraction of reference"
    ]
    assert df.loc[df["Drug"] == "A", "Fraction of reference"].iloc[0] == pytest.approx(1.2)
    assert df.loc[df["Drug"] == "B", "Fraction of reference"].iloc[0] == pytest.approx(0.5)


def test_compare_side_by_side():
    scale = MTDScale({"Adriamycin": 112.0, "Herceptin": 837000.0})
    ours = {"Adriamycin": 123.0, "Herceptin": 167400.0}
    lit = {"Adriamycin": 1.00, "Herceptin": 0.20}
    df = scale.compare(ours, lit)
    assert set(df.columns) == {
        "Drug", "Our optimum (fraction)", "Literature (fraction)", "Delta"
    }
    row_adri = df[df["Drug"] == "Adriamycin"].iloc[0]
    assert row_adri["Our optimum (fraction)"] == pytest.approx(123.0 / 112.0)
    assert row_adri["Literature (fraction)"] == 1.00
    assert row_adri["Delta"] == pytest.approx(123.0 / 112.0 - 1.00)


def test_missing_literature_entry_yields_none_delta():
    scale = MTDScale({"A": 100.0, "B": 200.0})
    df = scale.compare({"A": 50.0, "B": 100.0}, {"A": 0.6})  # B missing
    row_b = df[df["Drug"] == "B"].iloc[0]
    # pandas coerces None -> NaN in numeric columns
    assert pd.isna(row_b["Literature (fraction)"])
    assert pd.isna(row_b["Delta"])
