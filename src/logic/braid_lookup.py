# src/logic/braid_lookup.py
"""
Query engine for the pre-fitted BRAID surface database.
All functions are pure (no Streamlit imports) so they can be tested independently.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Default path — override via BRAID_PATH env var or argument ────────────────
_DEFAULT_BRAID = Path("C:/Max/Github/synergy_docker/braid_surface_prediction/data")
_DEFAULT_DATA  = Path("C:/Max/Github/synergy_docker/data")


def load_braid_index(braid_path=None):
    """
    Load braid_labels_qc.parquet.
    Returns a DataFrame with 216,846 rows, one per fitted surface.
    Decorated with @st.cache_data in the view layer.
    """
    p = Path(braid_path or _DEFAULT_BRAID) / "braid_labels_qc.parquet"
    if not p.exists():
        raise FileNotFoundError(f"braid_labels_qc.parquet not found at {p}")
    df = pd.read_parquet(p)
    # Normalise drug/cell names to UPPER for case-insensitive search
    df["_drug1_up"]  = df["drug1"].str.upper().str.strip()
    df["_drug2_up"]  = df["drug2"].str.upper().str.strip()
    df["_cell_up"]   = df["cell_line"].str.upper().str.strip()
    return df


def get_available_options(index_df):
    """
    Returns (sorted_drugs, sorted_cell_lines) for selectboxes.
    Drug list is the union of drug1 and drug2 columns.
    """
    drugs = sorted(set(index_df["drug1"].str.strip().tolist()) |
                   set(index_df["drug2"].str.strip().tolist()))
    cells = sorted(index_df["cell_line"].str.strip().unique().tolist())
    return drugs, cells


def query_surface(index_df, drug1, drug2, cell_line):
    """
    Case-insensitive lookup of (drug1, drug2, cell_line).
    Tries both drug orders (A+B and B+A — combination is symmetric).
    Returns a dict of parameters + metadata, or None if not found.
    """
    d1u = drug1.upper().strip()
    d2u = drug2.upper().strip()
    cu  = cell_line.upper().strip()

    mask = (
        (index_df["_drug1_up"] == d1u) &
        (index_df["_drug2_up"] == d2u) &
        (index_df["_cell_up"]  == cu)
    )
    if not mask.any():
        # Try swapped order
        mask = (
            (index_df["_drug1_up"] == d2u) &
            (index_df["_drug2_up"] == d1u) &
            (index_df["_cell_up"]  == cu)
        )

    if not mask.any():
        return None

    row = index_df[mask].iloc[0]
    return {
        "drug1":      row["drug1"],
        "drug2":      row["drug2"],
        "cell_line":  row["cell_line"],
        "BlockID":    row["BlockID"],
        "source":     row.get("source", "Unknown"),
        "EC50_1":     float(row["EC50_1"]),
        "h1":         float(row["h1"]),
        "EC50_2":     float(row["EC50_2"]),
        "h2":         float(row["h2"]),
        "E0":         float(row["E0"]),
        "Einf":       float(row["Einf"]),
        "kappa":      float(row["kappa"]),
        "r2_braid":   float(row["r2_braid"]),
        "rmse_braid": float(row["rmse_braid"]),
        "DepMap_ID":  row.get("DepMap_ID", None),
    }


def reconstruct_surface(params, n_points=60):
    """
    Reconstruct the BRAID viability surface from fitted parameters.
    Dose range: 0 to 4*EC50 for each drug.

    Returns:
        d1_arr : 1-D array of drug1 doses (length n_points)
        d2_arr : 1-D array of drug2 doses (length n_points)
        V_grid : 2-D array [n_points, n_points] of viability %
    """
    d1 = np.linspace(0, 4.0 * params["EC50_1"], n_points)
    d2 = np.linspace(0, 4.0 * params["EC50_2"], n_points)
    D1, D2 = np.meshgrid(d1, d2)

    h1, h2       = params["h1"],    params["h2"]
    EC1, EC2     = params["EC50_1"], params["EC50_2"]
    E0, Einf     = params["E0"],    params["Einf"]
    kappa        = params["kappa"]

    with np.errstate(over="ignore", invalid="ignore"):
        T1    = np.clip((D1 / EC1) ** h1, 0, 1e6)
        T2    = np.clip((D2 / EC2) ** h2, 0, 1e6)
        denom = np.clip(1.0 + T1 + T2 + kappa * T1 * T2, 1e-9, None)
        V     = Einf + (E0 - Einf) / denom

    return d1, d2, V


def kappa_label(kappa):
    """Returns (label_text, hex_color) based on kappa value."""
    if kappa > 1:
        return "Synergistic", "#27ae60"
    if kappa < -1:
        return "Antagonistic", "#e74c3c"
    return "Additive", "#2980b9"


def load_raw_checkerboard(block_id, data_path=None):
    """
    Load raw measured viability rows for a given BlockID.
    Tries parquet first (fast), falls back to CSV scan.
    Returns DataFrame with [ConcRow, ConcCol, Response] or None.
    """
    dp = Path(data_path or _DEFAULT_DATA)

    # Fast path: partitioned parquet
    pq_dir = dp / "checkerboard_drugcomb.parquet"
    if pq_dir.exists():
        try:
            part = pq_dir / f"BlockID={block_id}"
            if part.exists():
                return pd.read_parquet(part)[["ConcRow", "ConcCol", "Response"]]
        except Exception:
            pass

    # Slow path: scan the full CSV (only if parquet not available)
    csv_path = dp / "checkerboard_drugcomb.csv"
    if not csv_path.exists():
        return None
    try:
        chunks = pd.read_csv(csv_path, chunksize=50_000,
                             usecols=["BlockID", "ConcRow", "ConcCol", "Response"])
        rows = []
        for chunk in chunks:
            hit = chunk[chunk["BlockID"] == block_id]
            if not hit.empty:
                rows.append(hit)
        if rows:
            return pd.concat(rows)[["ConcRow", "ConcCol", "Response"]].reset_index(drop=True)
    except Exception:
        pass
    return None


def _braid_viability(d1, d2, params):
    """Scalar BRAID viability at a single (d1, d2) dose point."""
    EC1, EC2 = params["EC50_1"], params["EC50_2"]
    h1,  h2  = params["h1"],    params["h2"]
    E0, Einf = params["E0"],    params["Einf"]
    kappa    = params["kappa"]
    T1 = float(np.clip((d1 / EC1) ** h1, 0, 1e6)) if d1 > 0 else 0.0
    T2 = float(np.clip((d2 / EC2) ** h2, 0, 1e6)) if d2 > 0 else 0.0
    denom = max(1.0 + T1 + T2 + kappa * T1 * T2, 1e-9)
    return float(Einf + (E0 - Einf) / denom)


def _mono_dose_for_target(EC, h, E0, Einf, target_v):
    """
    Dose of a single drug required to reach target_v viability.
    Returns None when target_v < Einf (impossible for monotherapy).
    """
    if target_v <= Einf:
        return None
    ratio = (E0 - Einf) / (target_v - Einf) - 1.0
    if ratio <= 0:
        return None
    return float(EC * (ratio ** (1.0 / h)))


def optimize_braid_dose(params, objective="min_dose",
                        target_viability=50.0,
                        d1_max_mult=4.0, d2_max_mult=4.0,
                        dose_weight=(1.0, 1.0)):
    """
    Optimize drug doses on a BRAID surface using scipy.

    objective:
        "min_dose"  — minimise w1*d1 + w2*d2  s.t. V(d1,d2) ≤ target_viability
        "max_kill"  — minimise V(d1,d2)        s.t. d1 ≤ d1_max, d2 ≤ d2_max

    Returns a dict with optimal doses and benchmark comparisons, or None on failure.
    """
    from scipy.optimize import minimize as _minimize

    EC1, EC2 = params["EC50_1"], params["EC50_2"]
    E0, Einf = params["E0"],    params["Einf"]
    d1_max   = d1_max_mult * EC1
    d2_max   = d2_max_mult * EC2
    w1, w2   = dose_weight

    def obj_min_dose(d):
        return w1 * d[0] + w2 * d[1]

    def obj_max_kill(d):
        return _braid_viability(d[0], d[1], params)

    bounds = [(0.0, d1_max), (0.0, d2_max)]

    if objective == "min_dose":
        # Check feasibility: can the surface reach target_viability at all?
        v_corner = _braid_viability(d1_max, d2_max, params)
        if v_corner > target_viability:
            return {"feasible": False,
                    "message": f"Target {target_viability:.1f}% viability is not achievable "
                               f"within {d1_max_mult}×EC50 dose range "
                               f"(minimum achievable: {v_corner:.1f}%)."}

        constraint = {"type": "ineq",
                      "fun": lambda d: target_viability - _braid_viability(d[0], d[1], params)}

        starts = [
            (EC1,       EC2),
            (2.0 * EC1, EC2),
            (EC1,       2.0 * EC2),
            (2.0 * EC1, 2.0 * EC2),
            (0.5 * EC1, 2.0 * EC2),
            (2.0 * EC1, 0.5 * EC2),
        ]
        best = None
        for x0 in starts:
            res = _minimize(obj_min_dose, x0, method="SLSQP", bounds=bounds,
                            constraints=[constraint],
                            options={"ftol": 1e-10, "maxiter": 2000})
            v_at = _braid_viability(res.x[0], res.x[1], params)
            if v_at <= target_viability + 1.0:          # allow 1% slack
                if best is None or res.fun < best.fun:
                    best = res

        if best is None:
            return {"feasible": False,
                    "message": "Optimizer could not find a feasible solution. "
                               "Try increasing the max dose range."}
        opt_d1, opt_d2 = best.x

    else:  # max_kill
        res = _minimize(obj_max_kill,
                        [d1_max * 0.5, d2_max * 0.5],
                        method="L-BFGS-B", bounds=bounds,
                        options={"ftol": 1e-12, "maxiter": 2000})
        opt_d1, opt_d2 = res.x

    opt_v = _braid_viability(opt_d1, opt_d2, params)

    # Monotherapy benchmarks: what dose of each drug alone hits the same viability?
    mono1 = _mono_dose_for_target(EC1, params["h1"], E0, Einf, opt_v)
    mono2 = _mono_dose_for_target(EC2, params["h2"], E0, Einf, opt_v)

    # Dose-reduction index: ratio of combo dose to monotherapy dose
    dri1 = (mono1 / opt_d1) if (mono1 and opt_d1 > 0) else None
    dri2 = (mono2 / opt_d2) if (mono2 and opt_d2 > 0) else None

    return {
        "feasible":    True,
        "d1_opt":      float(opt_d1),
        "d2_opt":      float(opt_d2),
        "viability":   float(opt_v),
        "mono1_equiv": float(mono1) if mono1 is not None else None,
        "mono2_equiv": float(mono2) if mono2 is not None else None,
        "dri1":        float(dri1) if dri1 is not None else None,
        "dri2":        float(dri2) if dri2 is not None else None,
        "d1_max":      float(d1_max),
        "d2_max":      float(d2_max),
        "objective":   objective,
        "target_v":    float(target_viability),
    }


def build_checkerboard_parquet(data_path=None):
    """
    One-time utility: converts checkerboard_drugcomb.csv to a BlockID-partitioned
    parquet directory for fast per-block lookup (~30 seconds, ~200 MB on disk).
    Run from command line: python -c "from logic.braid_lookup import build_checkerboard_parquet; build_checkerboard_parquet()"
    """
    dp  = Path(data_path or _DEFAULT_DATA)
    src = dp / "checkerboard_drugcomb.csv"
    dst = dp / "checkerboard_drugcomb.parquet"
    if dst.exists():
        print(f"Already exists: {dst}")
        return
    print(f"Converting {src} → partitioned parquet (this takes ~30s) ...")
    df = pd.read_csv(src, usecols=["BlockID", "ConcRow", "ConcCol", "Response"])
    df.to_parquet(dst, partition_cols=["BlockID"], index=False)
    print(f"Done: {dst}")
