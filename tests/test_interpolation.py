"""Unit tests for logic.interpolation.interpolate_to_n_rows."""
import numpy as np
import pandas as pd
import pytest

from logic.interpolation import interpolate_to_n_rows


# ---------------------------------------------------------------------------
# Shape / identity / trivial cases
# ---------------------------------------------------------------------------

def test_returns_copy_when_n_equals_len():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    out = interpolate_to_n_rows(df, n_target=3)
    assert len(out) == 3
    assert out is not df
    np.testing.assert_array_equal(out["a"].values, df["a"].values)


def test_n_target_less_than_len_raises():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="must be >="):
        interpolate_to_n_rows(df, n_target=2)


def test_single_row_raises():
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(ValueError, match="at least 2"):
        interpolate_to_n_rows(df, n_target=5)


# ---------------------------------------------------------------------------
# Row-index time axis (the common case — no explicit time column)
# ---------------------------------------------------------------------------

def test_linear_data_preserved_under_interpolation():
    """If the source column is already perfectly linear in row-index, the
    interpolated column must also be linear — i.e., interpolation introduces
    no error on a straight line."""
    df = pd.DataFrame({"y": np.linspace(0.0, 12.0, 13)})  # slope 1, intercept 0
    out = interpolate_to_n_rows(df, n_target=15)
    assert len(out) == 15
    expected = np.linspace(0.0, 12.0, 15)
    np.testing.assert_allclose(out["y"].values, expected, atol=1e-12)


def test_endpoints_are_preserved():
    """First and last source rows must equal first and last output rows."""
    df = pd.DataFrame({"a": [1.0, 5.0, 2.0, 8.0], "b": [10.0, 20.0, 15.0, 30.0]})
    out = interpolate_to_n_rows(df, n_target=10)
    assert out["a"].iloc[0] == df["a"].iloc[0]
    assert out["a"].iloc[-1] == df["a"].iloc[-1]
    assert out["b"].iloc[0] == df["b"].iloc[0]
    assert out["b"].iloc[-1] == df["b"].iloc[-1]


def test_column_order_and_count_preserved():
    df = pd.DataFrame({"z": [1.0, 2.0], "a": [10.0, 20.0], "m": [100.0, 200.0]})
    out = interpolate_to_n_rows(df, n_target=5)
    assert list(out.columns) == ["z", "a", "m"]
    assert len(out) == 5


# ---------------------------------------------------------------------------
# Named time column
# ---------------------------------------------------------------------------

def test_named_time_col_used_as_interpolation_axis():
    """If time_col is supplied, it must be used as the x-axis and also
    appear as uniform spacing in the output."""
    df = pd.DataFrame({
        "Day": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [0.0, 10.0, 20.0, 30.0, 40.0],  # slope 10
    })
    out = interpolate_to_n_rows(df, n_target=9, time_col="Day")
    assert len(out) == 9
    np.testing.assert_allclose(out["Day"].values, np.linspace(1.0, 5.0, 9))
    # y = 10 * (Day - 1), so interpolated y should match
    expected_y = 10.0 * (out["Day"].values - 1.0)
    np.testing.assert_allclose(out["y"].values, expected_y, atol=1e-12)


def test_missing_time_col_raises():
    df = pd.DataFrame({"y": [1.0, 2.0]})
    with pytest.raises(KeyError):
        interpolate_to_n_rows(df, n_target=4, time_col="Day")


def test_non_monotone_time_col_raises():
    df = pd.DataFrame({
        "Day": [1.0, 3.0, 2.0, 4.0],  # not sorted
        "y": [1.0, 2.0, 3.0, 4.0],
    })
    with pytest.raises(ValueError, match="strictly increasing"):
        interpolate_to_n_rows(df, n_target=8, time_col="Day")


# ---------------------------------------------------------------------------
# Non-numeric columns
# ---------------------------------------------------------------------------

def test_non_numeric_columns_filled_by_nearest_neighbor():
    df = pd.DataFrame({
        "label": ["A", "B", "C"],
        "y": [0.0, 10.0, 20.0],
    })
    out = interpolate_to_n_rows(df, n_target=5)
    # t_new = [0, 1, 2, 3, 4] scaled to [0, 4] original → [0, 1, 2, 3, 4]
    # Wait: source t = [0, 1, 2], target = linspace(0, 2, 5) = [0, 0.5, 1, 1.5, 2]
    # Nearest original row for each target: 0, 0 or 1, 1, 1 or 2, 2
    # Tie-breaking via argmin gives the lower index.
    assert list(out["label"]) == ["A", "A", "B", "B", "C"]
    np.testing.assert_allclose(out["y"].values, [0.0, 5.0, 10.0, 15.0, 20.0])


# ---------------------------------------------------------------------------
# The paper scenario: 13 -> 15 on ior1 data
# ---------------------------------------------------------------------------

def test_paper_scenario_13_to_15_matches_expected_shape():
    """Smoke test — exercise the Ding et al. use case (13 daily rows → 15)."""
    df = pd.DataFrame({
        "Adriamycin": np.linspace(56.2, 113.0, 13),
        "Gemcitabine": np.linspace(81900.0, 164000.0, 13),
        "Cisplatin": np.linspace(1530.0, 3220.0, 13),
        "Herceptin": np.linspace(382000.0, 908000.0, 13),
        "Rat_TumorSize": np.linspace(1.22, 1.96, 13),
    })
    out = interpolate_to_n_rows(df, n_target=15)
    assert len(out) == 15
    # Endpoints preserved
    assert out["Rat_TumorSize"].iloc[0] == pytest.approx(1.22)
    assert out["Rat_TumorSize"].iloc[-1] == pytest.approx(1.96)
    # 15 rows → 15 polynomial params is an exactly-identified system
    # (this is what makes fitnlm / LM happy)
    assert len(out) >= 15
