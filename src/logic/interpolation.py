"""
Dataset interpolation — densify a measurement table to a larger row count.

Motivated by the AI-PRS paper (Ding et al., Adv. Therap. 2020), which
interpolates its 13 daily rat measurements to 15 rows so the 15-coefficient
quadratic PRS is exactly identifiable by Matlab `fitnlm`. More generally,
any time the user wants n_obs >= n_params for OLS / fitnlm to run cleanly,
this is the pre-processing step.

Linear interpolation via numpy. Numeric columns are interpolated along a
uniform grid of the time axis; non-numeric columns are filled by nearest
neighbor so labels / categorical values don't get mangled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def interpolate_to_n_rows(
    df: pd.DataFrame,
    n_target: int,
    time_col: str | None = None,
) -> pd.DataFrame:
    """Return a new DataFrame with `n_target` rows produced by linear
    interpolation of `df` along its time axis.

    Parameters
    ----------
    df : pd.DataFrame
        Source table. Must have at least 2 rows and 1 numeric column.
    n_target : int
        Desired row count. Must be >= len(df); equal returns a copy.
    time_col : str, optional
        Name of the column to use as the interpolation x-axis. Values must
        be numeric and strictly monotonically increasing. If omitted, the
        row index is used (implicitly assumes rows are equi-spaced in time).

    Returns
    -------
    pd.DataFrame
        Same column order and dtypes as `df`. Numeric columns are linearly
        interpolated onto `n_target` uniformly-spaced points across the
        time range. Non-numeric columns are filled by nearest-neighbor
        from the original rows.

    Raises
    ------
    ValueError
        If the source has fewer than 2 rows, n_target < len(df), time_col
        is not strictly increasing, or no numeric columns exist to
        interpolate.
    KeyError
        If `time_col` is provided but not a column of `df`.
    """
    if len(df) < 2:
        raise ValueError(f"Need at least 2 rows to interpolate; got {len(df)}.")
    if n_target < len(df):
        raise ValueError(
            f"n_target={n_target} must be >= len(df)={len(df)}."
        )
    if n_target == len(df):
        return df.copy().reset_index(drop=True)

    if time_col is not None:
        if time_col not in df.columns:
            raise KeyError(f"time_col '{time_col}' not in dataframe columns")
        t_orig = df[time_col].values.astype(float)
        if not np.all(np.diff(t_orig) > 0):
            raise ValueError(
                f"time_col '{time_col}' must be strictly increasing."
            )
        numeric_cols = [
            c for c in df.columns
            if c != time_col and pd.api.types.is_numeric_dtype(df[c])
        ]
    else:
        t_orig = np.arange(len(df), dtype=float)
        numeric_cols = [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]

    if not numeric_cols and time_col is None:
        raise ValueError("No numeric columns found to interpolate.")

    t_new = np.linspace(t_orig[0], t_orig[-1], n_target)

    out: dict[str, np.ndarray] = {}
    if time_col is not None:
        out[time_col] = t_new
    for col in numeric_cols:
        y_orig = df[col].values.astype(float)
        out[col] = np.interp(t_new, t_orig, y_orig)

    non_numeric = [
        c for c in df.columns
        if c != time_col and c not in numeric_cols
    ]
    if non_numeric:
        nearest_idx = np.array(
            [int(np.argmin(np.abs(t_orig - t))) for t in t_new]
        )
        for col in non_numeric:
            out[col] = df[col].iloc[nearest_idx].values

    return pd.DataFrame({c: out[c] for c in df.columns}).reset_index(drop=True)
