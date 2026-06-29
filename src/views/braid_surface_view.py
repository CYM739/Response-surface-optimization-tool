# src/views/braid_surface_view.py
"""
Unified BRAID Surface tab — merges Lookup + Prediction into one flow.

1. User selects Drug 1, Drug 2, Cell Line → clicks Search
2. If found in DB → show fitted surface + offer neural network comparison
3. If NOT found  → offer prediction with v19 model
4. When both exist → side-by-side comparison with delta metrics
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path
import os

from logic.braid_lookup import (
    load_braid_index,
    get_available_options,
    query_surface,
    reconstruct_surface,
    kappa_label,
    load_raw_checkerboard,
    optimize_braid_dose,
)

# ── Paths ────────────────────────────────────────────────────────────────────
_SYNERGY_ROOT = Path(os.environ.get(
    "SYNERGY_ROOT", "C:/Max/Github/synergy_docker"))
_BRAID_PATH = Path(os.environ.get("BRAID_PATH",
    str(_SYNERGY_ROOT / "braid_surface_prediction" / "data")))
_DATA_PATH = Path(os.environ.get("DATA_PATH",
    str(_SYNERGY_ROOT / "data")))


# ── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading BRAID surface index...")
def _cached_load_index(braid_path_str: str):
    return load_braid_index(braid_path_str)


@st.cache_resource(show_spinner="Loading BRAID prediction model (one-time)...")
def _load_engine(checkpoint):
    from logic.braid_predict import BRAIDPredictionEngine
    return BRAIDPredictionEngine(checkpoint=checkpoint)


@st.cache_data(show_spinner=False)
def _load_cell_name_map():
    si_path = _DATA_PATH / "sample_info.csv"
    if si_path.exists():
        si = pd.read_csv(si_path, usecols=["DepMap_ID", "stripped_cell_line_name"],
                         low_memory=False)
        return dict(zip(si["DepMap_ID"], si["stripped_cell_line_name"]))
    return {}


@st.cache_data(show_spinner=False)
def _build_drug_mono_defaults(_braid_path_str: str):
    """Compute per-drug median mono params from the QC database for fallback."""
    qc = pd.read_parquet(Path(_braid_path_str) / "braid_labels_qc.parquet")
    d1 = qc.groupby("drug1")[["EC50_1", "h1"]].median().rename(
        columns={"EC50_1": "EC50", "h1": "h"})
    d2 = qc.groupby("drug2")[["EC50_2", "h2"]].median().rename(
        columns={"EC50_2": "EC50", "h2": "h"})
    combined = pd.concat([d1, d2]).groupby(level=0).median()
    return combined


# ── Main render ──────────────────────────────────────────────────────────────

def render():
    st.header("🔬 BRAID Surface")
    st.caption(
        "Search 216,846 pre-fitted drug combination surfaces from DrugComb. "
        "If no fitted data exists, predict the surface with a trained neural network (v19)."
    )

    # ── 1. Load index ────────────────────────────────────────────────────────
    try:
        index_df = _cached_load_index(str(_BRAID_PATH))
    except FileNotFoundError as e:
        st.error(f"BRAID index not found: {e}")
        return

    drugs, db_cell_lines = get_available_options(index_df)
    cell_names = _load_cell_name_map()

    # Merge cell lines: QC database (by name) + all CCLE cells (for prediction)
    # Build unified list: "CellName" for QC cells, "CellName (ACH-xxx, predict only)" for CCLE-only
    ccle_cells_by_name = {v: k for k, v in cell_names.items()
                          if isinstance(v, str)}  # name → DepMap_ID
    db_set = set(c.upper().strip() for c in db_cell_lines)
    all_cell_options = sorted(set(db_cell_lines))
    for name, depmap in ccle_cells_by_name.items():
        if name.upper().strip() not in db_set:
            all_cell_options.append(f"{name} (predict only)")
    all_cell_options = sorted(all_cell_options)

    # ── 2. Search controls ───────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    drug1 = col1.selectbox("Drug 1", drugs, key="bs_drug1")
    drug2 = col2.selectbox("Drug 2", drugs, key="bs_drug2")
    cell_selection = col3.selectbox("Cell Line", all_cell_options, key="bs_cell")
    search = col4.button("Search", use_container_width=True)

    # Resolve display label back to the raw cell line name
    predict_only = cell_selection.endswith("(predict only)")
    cell_line = cell_selection.replace(" (predict only)", "").strip()

    if drug1 == drug2:
        st.warning("Drug 1 and Drug 2 must be different.")
        return

    # ── 3. Query DB ──────────────────────────────────────────────────────────
    if search:
        # Skip DB query for predict-only cells (not in fitted database)
        result = None if predict_only else query_surface(index_df, drug1, drug2, cell_line)
        st.session_state["bs_lookup"] = result
        st.session_state["bs_query"] = (drug1, drug2, cell_line)
        st.session_state["bs_searched"] = True
        # Clear previous prediction when doing a new search
        st.session_state.pop("bs_prediction", None)

    if not st.session_state.get("bs_searched", False):
        st.info("Select Drug 1, Drug 2, and Cell Line, then click **Search**.")
        return

    lookup = st.session_state.get("bs_lookup")
    query = st.session_state.get("bs_query")

    d1_label = query[0] if query else drug1
    d2_label = query[1] if query else drug2
    cl_label = query[2] if query else cell_line

    # ── 4. Branch: found vs not found ────────────────────────────────────────
    if lookup is not None:
        _render_found(lookup, d1_label, d2_label, cl_label)
    else:
        _render_not_found(d1_label, d2_label, cl_label)


# ═════════════════════════════════════════════════════════════════════════════
#  BRANCH A: Surface found in database
# ═════════════════════════════════════════════════════════════════════════════

def _render_found(result, d1_label, d2_label, cl_label):
    """Show the fitted surface + offer neural network comparison."""

    # ── Parameters + kappa badge ─────────────────────────────────────────────
    p_col, b_col = st.columns([3, 2])

    with p_col:
        st.subheader("Fitted BRAID Parameters")
        st.table({
            "Parameter": [
                "EC50_1", "h1 (Hill slope)",
                "EC50_2", "h2 (Hill slope)",
                "E0 (baseline %)", "Einf (max effect %)",
                "kappa", "R²", "RMSE",
            ],
            "Value": [
                f"{result['EC50_1']:.4g}", f"{result['h1']:.3f}",
                f"{result['EC50_2']:.4g}", f"{result['h2']:.3f}",
                f"{result['E0']:.1f}", f"{result['Einf']:.1f}",
                f"{result['kappa']:.3f}", f"{result['r2_braid']:.3f}",
                f"{result['rmse_braid']:.2f}",
            ],
        })
        st.caption(
            f"Source: **{result['source']}** · BlockID: `{result['BlockID']}` · "
            f"Drug order: **{result['drug1']}** + **{result['drug2']}**"
        )

    with b_col:
        _render_kappa_badge(result["kappa"])

    st.divider()

    # ── Reconstruct surface ──────────────────────────────────────────────────
    d1_arr, d2_arr, V_grid = reconstruct_surface(result, n_points=60)

    # ── Visualisation tabs ───────────────────────────────────────────────────
    vt1, vt2, vt3 = st.tabs(["3D Surface", "2D Heatmap", "Raw Data"])

    with vt1:
        _plot_3d_surface(d1_arr, d2_arr, V_grid, d1_label, d2_label, cl_label,
                         title=f"{d1_label} + {d2_label} in {cl_label}",
                         key_suffix="fitted")

    with vt2:
        _plot_2d_heatmap(d1_arr, d2_arr, V_grid, d1_label, d2_label, cl_label)

    with vt3:
        _render_raw_data(result, d1_arr, d2_arr, V_grid, d1_label, d2_label)

    # ── Dose optimisation ────────────────────────────────────────────────────
    _render_dose_optimisation(result, d1_label, d2_label)

    # ── Offer neural network comparison ──────────────────────────────────────
    st.divider()
    _render_nn_comparison_offer(result, d1_label, d2_label, cl_label)


# ═════════════════════════════════════════════════════════════════════════════
#  BRANCH B: Surface NOT found — offer prediction
# ═════════════════════════════════════════════════════════════════════════════

def _render_not_found(d1_label, d2_label, cl_label):
    """No fitted surface — offer neural network prediction."""
    st.warning(
        f"No fitted surface found for **{d1_label}** + **{d2_label}** "
        f"in **{cl_label}**."
    )

    st.info(
        "You can predict this combination's response surface using the trained "
        "v19 neural network model. The model requires monotherapy parameters "
        "(EC50 and Hill slope) for each drug."
    )

    # ── Model checkpoint selector ────────────────────────────────────────────
    checkpoint = st.selectbox(
        "Model checkpoint",
        options=[
            "best_v19_cold_drug.pth",
            "best_v19_random.pth",
            "best_v19_cold_cell.pth",
        ],
        index=0,
        help=(
            "cold_drug: best for new drugs. "
            "cold_cell: best for new cell lines. "
            "random: general-purpose."
        ),
        key="bs_ckpt",
    )

    # ── Load engine ──────────────────────────────────────────────────────────
    try:
        engine = _load_engine(checkpoint)
    except Exception as e:
        st.error(f"Failed to load prediction model: {e}")
        return

    # ── Resolve cell line to DepMap_ID ────────────────────────────────────────
    cell_names = _load_cell_name_map()
    name_to_id = {v: k for k, v in cell_names.items()}
    cell_id = name_to_id.get(cl_label, cl_label)

    avail_cells = set(engine.get_available_cells())
    if cell_id not in avail_cells:
        st.warning(
            f"Cell line **{cl_label}** ({cell_id}) is not available in the "
            "model's feature set (no CCLE expression data). Prediction is not possible."
        )
        return

    # ── Mono params: try exact lookup, then per-drug median fallback ─────────
    mono_lookup = engine.lookup_mono_params(d1_label, d2_label, cell_id)

    if mono_lookup is None:
        # Fallback: per-drug median from all surfaces in QC
        try:
            drug_defaults = _build_drug_mono_defaults(str(_BRAID_PATH))
            ec1 = float(drug_defaults.loc[d1_label, "EC50"]) if d1_label in drug_defaults.index else 1.0
            h1_def = float(drug_defaults.loc[d1_label, "h"]) if d1_label in drug_defaults.index else 1.0
            ec2 = float(drug_defaults.loc[d2_label, "EC50"]) if d2_label in drug_defaults.index else 1.0
            h2_def = float(drug_defaults.loc[d2_label, "h"]) if d2_label in drug_defaults.index else 1.0
            mono_lookup = {"EC50_1": ec1, "h1": h1_def, "EC50_2": ec2, "h2": h2_def}
            fallback_source = "per-drug median"
        except Exception:
            mono_lookup = {"EC50_1": 1.0, "h1": 1.0, "EC50_2": 1.0, "h2": 1.0}
            fallback_source = "defaults"
    else:
        fallback_source = "exact match"

    _render_prediction_panel(
        engine, d1_label, d2_label, cl_label, cell_id,
        mono_lookup, fallback_source, checkpoint,
        ground_truth=None,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Neural network comparison (offered from Branch A)
# ═════════════════════════════════════════════════════════════════════════════

def _render_nn_comparison_offer(fitted_result, d1_label, d2_label, cl_label):
    """From the fitted-surface view, offer to compare with NN prediction."""
    st.subheader("Neural Network Comparison")
    st.caption(
        "Compare the curve-fitted result above with the v19 neural network prediction. "
        "This shows how well the model generalises to this combination."
    )

    checkpoint = st.selectbox(
        "Model checkpoint",
        options=[
            "best_v19_cold_drug.pth",
            "best_v19_random.pth",
            "best_v19_cold_cell.pth",
        ],
        index=0,
        key="bs_cmp_ckpt",
    )

    compare_btn = st.button("Run Neural Network Prediction", key="bs_compare")

    if compare_btn:
        try:
            engine = _load_engine(checkpoint)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

        cell_names = _load_cell_name_map()
        name_to_id = {v: k for k, v in cell_names.items()}
        cell_id = name_to_id.get(cl_label, cl_label)

        # Use the exact mono params from the fitted surface
        mono_params = {
            "EC50_1": fitted_result["EC50_1"],
            "h1": fitted_result["h1"],
            "EC50_2": fitted_result["EC50_2"],
            "h2": fitted_result["h2"],
        }

        with st.spinner("Running inference..."):
            try:
                pred = engine.predict_surface(
                    fitted_result["drug1"], fitted_result["drug2"],
                    cell_id, mono_params)
                st.session_state["bs_comparison"] = pred
            except Exception as e:
                import traceback
                st.error(f"Prediction failed: {e}")
                st.code(traceback.format_exc())
                return

    pred = st.session_state.get("bs_comparison")
    if pred is None:
        return

    # ── Delta metrics ────────────────────────────────────────────────────────
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("kappa (fitted)", f"{fitted_result['kappa']:.3f}")
    mc2.metric("kappa (predicted)", f"{pred['kappa']:.3f}")
    mc3.metric("Delta kappa", f"{pred['kappa'] - fitted_result['kappa']:.3f}")
    mc4.metric("Delta E0", f"{pred['E0'] - fitted_result['E0']:.1f}")
    mc5.metric("Delta Einf", f"{pred['Einf'] - fitted_result['Einf']:.1f}")

    # ── Side-by-side surfaces ────────────────────────────────────────────────
    d1_fit, d2_fit, V_fit = reconstruct_surface(fitted_result, n_points=60)

    sc1, sc2 = st.columns(2)
    with sc1:
        _plot_3d_surface(d1_fit, d2_fit, V_fit, d1_label, d2_label, cl_label,
                         title="Fitted (ground truth)", height=420,
                         key_suffix="cmp_fitted")
    with sc2:
        _plot_3d_surface(pred["d1_arr"], pred["d2_arr"], pred["V_grid"],
                         d1_label, d2_label, cl_label,
                         title=f"Predicted (v19 — {checkpoint})", height=420,
                         key_suffix="cmp_pred")


# ═════════════════════════════════════════════════════════════════════════════
#  Shared prediction panel (used by Branch B)
# ═════════════════════════════════════════════════════════════════════════════

def _render_prediction_panel(engine, d1_label, d2_label, cl_label, cell_id,
                             mono_defaults, fallback_source, checkpoint,
                             ground_truth=None):
    """Mono param inputs + predict button + results display."""

    # Track selection changes to reset mono params
    _sel_key = f"bs|{d1_label}|{d2_label}|{cell_id}"
    if st.session_state.get("_bs_sel_key") != _sel_key:
        st.session_state["_bs_sel_key"] = _sel_key
        st.session_state["bs_ec50_1"] = mono_defaults["EC50_1"]
        st.session_state["bs_h1"] = mono_defaults["h1"]
        st.session_state["bs_ec50_2"] = mono_defaults["EC50_2"]
        st.session_state["bs_h2"] = mono_defaults["h2"]

    with st.expander("Monotherapy Parameters (EC50 & Hill slope)", expanded=True):
        if fallback_source == "exact match":
            st.success("Auto-filled from the fitted surface database (exact match).")
        elif fallback_source == "per-drug median":
            st.info(
                "Auto-filled with **per-drug median** values from the database "
                "(no exact match for this combination). Adjust if you have experimental data."
            )
        else:
            st.warning("Using default values. Enter your experimental monotherapy parameters.")

        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(f"**{d1_label}**")
            ec50_1 = st.number_input(
                "EC50_1", min_value=1e-12, format="%.4g", key="bs_ec50_1")
            h1 = st.number_input(
                "Hill slope h1", min_value=0.01, format="%.3f", key="bs_h1")
        with mc2:
            st.markdown(f"**{d2_label}**")
            ec50_2 = st.number_input(
                "EC50_2", min_value=1e-12, format="%.4g", key="bs_ec50_2")
            h2 = st.number_input(
                "Hill slope h2", min_value=0.01, format="%.3f", key="bs_h2")

    mono_params = {"EC50_1": ec50_1, "h1": h1, "EC50_2": ec50_2, "h2": h2}

    predict_btn = st.button("Predict Surface", type="primary", key="bs_predict")

    if predict_btn:
        with st.spinner("Running inference..."):
            try:
                result = engine.predict_surface(d1_label, d2_label, cell_id, mono_params)
                st.session_state["bs_prediction"] = result
            except Exception as e:
                import traceback
                st.error(f"Prediction failed: {e}")
                st.code(traceback.format_exc())
                return

    pred = st.session_state.get("bs_prediction")
    if pred is None:
        return

    # ── Results ──────────────────────────────────────────────────────────────
    cell_names = _load_cell_name_map()
    cl_display = cell_names.get(cell_id, cl_label)

    p_col, b_col = st.columns([3, 2])
    with p_col:
        st.subheader("Predicted BRAID Parameters")
        st.table({
            "Parameter": [
                "EC50_1 (input)", "h1 (input)",
                "EC50_2 (input)", "h2 (input)",
                "E0 (predicted)", "Einf (predicted)",
                "kappa (predicted)",
            ],
            "Value": [
                f"{pred['EC50_1']:.4g}", f"{pred['h1']:.3f}",
                f"{pred['EC50_2']:.4g}", f"{pred['h2']:.3f}",
                f"{pred['E0']:.1f}", f"{pred['Einf']:.1f}",
                f"{pred['kappa']:.3f}",
            ],
        })
        st.caption(f"Model: `{checkpoint}` · Cell: **{cl_display}** ({cell_id})")

    with b_col:
        _render_kappa_badge(pred["kappa"])

    st.divider()

    # ── Visualisation ────────────────────────────────────────────────────────
    vt1, vt2, vt3 = st.tabs(["3D Surface", "2D Heatmap", "Dose-Response Slices"])
    with vt1:
        _plot_3d_surface(pred["d1_arr"], pred["d2_arr"], pred["V_grid"],
                         d1_label, d2_label, cl_display,
                         title=f"Predicted: {d1_label} + {d2_label} in {cl_display}",
                         key_suffix="notfound_pred")
    with vt2:
        _plot_2d_heatmap(pred["d1_arr"], pred["d2_arr"], pred["V_grid"],
                         d1_label, d2_label, cl_display)
    with vt3:
        _render_dose_slices(pred, d1_label, d2_label)


# ═════════════════════════════════════════════════════════════════════════════
#  Shared UI components
# ═════════════════════════════════════════════════════════════════════════════

def _render_kappa_badge(kappa):
    label, color = kappa_label(kappa)
    st.markdown(
        f"""
        <div style="background:{color}22; border:2px solid {color};
                    border-radius:12px; padding:24px; text-align:center; margin-top:16px">
            <div style="font-size:1.8em; font-weight:bold">{label}</div>
            <div style="font-size:1.5em; font-weight:bold; margin-top:6px">
                kappa = {kappa:.3f}
            </div>
            <div style="font-size:0.82em; color:#666; margin-top:10px">
                kappa &gt; 1 &rarr; synergistic<br>
                &minus;1 &le; kappa &le; 1 &rarr; additive<br>
                kappa &lt; &minus;1 &rarr; antagonistic
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _plot_3d_surface(d1_arr, d2_arr, V_grid, d1_label, d2_label, cl_label,
                     title=None, height=540, key_suffix=""):
    # Dose axis scale toggle
    use_log = st.checkbox("Log-scale dose axes", value=False,
                          key=f"bs_log_{key_suffix}" if key_suffix else None,
                          help="Spreads out the low-dose region for steep Hill slopes")

    if use_log:
        # Replace zero with a small value, then log-transform
        eps1 = d1_arr[d1_arr > 0].min() / 10 if (d1_arr > 0).any() else 1e-6
        eps2 = d2_arr[d2_arr > 0].min() / 10 if (d2_arr > 0).any() else 1e-6
        x_plot = np.log10(np.where(d1_arr > 0, d1_arr, eps1))
        y_plot = np.log10(np.where(d2_arr > 0, d2_arr, eps2))
        x_title = f"{d1_label} dose (log10)"
        y_title = f"{d2_label} dose (log10)"
    else:
        x_plot = d1_arr
        y_plot = d2_arr
        x_title = f"{d1_label} dose"
        y_title = f"{d2_label} dose"

    fig = go.Figure(data=[go.Surface(
        x=x_plot, y=y_plot, z=V_grid,
        colorscale="RdBu", reversescale=True,
        colorbar=dict(title="Viability %", thickness=14),
        customdata=np.dstack(np.meshgrid(d1_arr, d2_arr)),
        hovertemplate=(
            f"{d1_label}: %{{customdata[0]:.3g}}<br>"
            f"{d2_label}: %{{customdata[1]:.3g}}<br>"
            "Viability: %{z:.1f}%<extra></extra>"
        ),
    )])
    fig.update_layout(
        scene=dict(
            xaxis_title=x_title,
            yaxis_title=y_title,
            zaxis_title="Viability (%)",
            zaxis=dict(range=[
                max(0, float(V_grid.min()) - 5),
                min(110, float(V_grid.max()) + 5),
            ]),
        ),
        title=title or f"{d1_label} + {d2_label} in {cl_label}",
        height=height,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_2d_heatmap(d1_arr, d2_arr, V_grid, d1_label, d2_label, cl_label):
    fig = go.Figure(data=go.Heatmap(
        x=d1_arr, y=d2_arr, z=V_grid,
        colorscale="RdBu", reversescale=True,
        colorbar=dict(title="Viability %"),
        hovertemplate=(
            f"{d1_label}: %{{x:.3g}}<br>"
            f"{d2_label}: %{{y:.3g}}<br>"
            "Viability: %{z:.1f}%<extra></extra>"
        ),
    ))
    fig.update_layout(
        xaxis_title=f"{d1_label} dose",
        yaxis_title=f"{d2_label} dose",
        title=f"Viability Heatmap — {d1_label} + {d2_label} in {cl_label}",
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_dose_slices(result, d1_label, d2_label):
    d1_arr = result["d1_arr"]
    d2_arr = result["d2_arr"]
    V_grid = result["V_grid"]
    n = len(d1_arr)
    slices = [0, n // 4, n // 2, 3 * n // 4, n - 1]

    sc1, sc2 = st.columns(2)
    with sc1:
        fig = go.Figure()
        for i in slices:
            fig.add_trace(go.Scatter(
                x=d1_arr, y=V_grid[i, :], mode="lines",
                name=f"{d2_label}={d2_arr[i]:.3g}"))
        fig.update_layout(
            title=f"{d1_label} dose-response at fixed {d2_label}",
            xaxis_title=f"{d1_label} dose", yaxis_title="Viability (%)",
            height=380)
        st.plotly_chart(fig, use_container_width=True)

    with sc2:
        fig = go.Figure()
        for i in slices:
            fig.add_trace(go.Scatter(
                x=d2_arr, y=V_grid[:, i], mode="lines",
                name=f"{d1_label}={d1_arr[i]:.3g}"))
        fig.update_layout(
            title=f"{d2_label} dose-response at fixed {d1_label}",
            xaxis_title=f"{d2_label} dose", yaxis_title="Viability (%)",
            height=380)
        st.plotly_chart(fig, use_container_width=True)


def _render_raw_data(result, d1_arr, d2_arr, V_grid, d1_label, d2_label):
    raw = load_raw_checkerboard(result["BlockID"], str(_DATA_PATH))
    if raw is not None and not raw.empty:
        st.caption(
            f"Raw measured checkerboard data for BlockID `{result['BlockID']}` "
            f"({len(raw)} data points)"
        )
        sc_col, tbl_col = st.columns([3, 2])
        with sc_col:
            fig_raw = go.Figure()
            fig_raw.add_trace(go.Surface(
                x=d1_arr, y=d2_arr, z=V_grid,
                colorscale="RdBu", reversescale=True,
                opacity=0.5, showscale=False, name="Fitted surface"))
            fig_raw.add_trace(go.Scatter3d(
                x=raw["ConcRow"].values, y=raw["ConcCol"].values,
                z=raw["Response"].values,
                mode="markers",
                marker=dict(size=4, color="black", opacity=0.8),
                name="Measured",
                hovertemplate=(
                    f"{d1_label}: %{{x:.3g}}<br>"
                    f"{d2_label}: %{{y:.3g}}<br>"
                    "Viability: %{z:.1f}%<extra></extra>"
                ),
            ))
            fig_raw.update_layout(
                scene=dict(
                    xaxis_title=f"{d1_label} dose",
                    yaxis_title=f"{d2_label} dose",
                    zaxis_title="Viability (%)",
                ),
                title="Fitted Surface + Raw Measurements",
                height=480, margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_raw, use_container_width=True)
        with tbl_col:
            st.dataframe(
                raw[["ConcRow", "ConcCol", "Response"]].rename(columns={
                    "ConcRow": f"{d1_label} dose",
                    "ConcCol": f"{d2_label} dose",
                    "Response": "Viability %",
                }).round(3),
                use_container_width=True, height=430,
            )
    else:
        st.info(
            "Raw checkerboard data not available for this BlockID. "
            "To enable fast lookup, run:\n\n"
            "```python\n"
            "from logic.braid_lookup import build_checkerboard_parquet\n"
            "build_checkerboard_parquet()\n"
            "```"
        )


def _render_dose_optimisation(result, d1_label, d2_label):
    st.divider()
    with st.expander("Dose Optimisation", expanded=False):
        st.caption(
            "Find the optimal drug doses directly from the BRAID surface using scipy."
        )

        oc1, oc2 = st.columns([1, 1])
        with oc1:
            objective = st.radio(
                "Objective",
                options=["min_dose", "max_kill"],
                format_func=lambda x: {
                    "min_dose": "Minimise total dose for a target kill",
                    "max_kill": "Maximise kill within a dose budget",
                }[x],
                key="bs_opt_objective",
            )
        with oc2:
            d1_max_mult = st.slider(
                f"Max dose ({d1_label}) — x EC50",
                min_value=1.0, max_value=10.0, value=4.0, step=0.5, key="bs_d1_mult")
            d2_max_mult = st.slider(
                f"Max dose ({d2_label}) — x EC50",
                min_value=1.0, max_value=10.0, value=4.0, step=0.5, key="bs_d2_mult")

        if objective == "min_dose":
            v_min = float(result["Einf"])
            v_base = float(result["E0"])
            target_default = round(max(v_min + 5, (v_min + v_base) * 0.5), 1)
            target_v = st.slider(
                "Target viability (%)",
                min_value=float(max(0, v_min + 1)), max_value=float(v_base),
                value=float(min(target_default, v_base - 1)),
                step=1.0, key="bs_target_v",
                help=f"Einf (floor) = {v_min:.1f}%  |  E0 (baseline) = {v_base:.1f}%",
            )
            wc1, wc2 = st.columns(2)
            w1 = wc1.number_input(f"Dose weight — {d1_label}", value=1.0,
                                  min_value=0.01, step=0.1, key="bs_w1")
            w2 = wc2.number_input(f"Dose weight — {d2_label}", value=1.0,
                                  min_value=0.01, step=0.1, key="bs_w2")
        else:
            target_v = None
            w1, w2 = 1.0, 1.0

        run_opt = st.button("Run Optimisation", key="bs_run_opt")

        if run_opt:
            with st.spinner("Optimising..."):
                opt = optimize_braid_dose(
                    result, objective=objective,
                    target_viability=target_v if target_v is not None else 50.0,
                    d1_max_mult=d1_max_mult, d2_max_mult=d2_max_mult,
                    dose_weight=(w1, w2))
            st.session_state["bs_opt_result"] = opt

        opt = st.session_state.get("bs_opt_result")
        if opt is None:
            st.stop()

        if not opt.get("feasible", True):
            st.warning(opt.get("message", "Optimisation failed."))
            st.stop()

        st.subheader("Optimal Doses")
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{d1_label} dose", f"{opt['d1_opt']:.4g}")
        m2.metric(f"{d2_label} dose", f"{opt['d2_opt']:.4g}")
        m3.metric("Predicted viability", f"{opt['viability']:.1f}%")

        st.subheader("Monotherapy Comparison")
        cmp1, cmp2 = st.columns(2)
        with cmp1:
            mono1 = opt.get("mono1_equiv")
            dri1 = opt.get("dri1")
            if mono1 is not None:
                st.metric(f"{d1_label} alone", f"{mono1:.4g}",
                          delta=f"{dri1:.1f}x dose reduction" if dri1 else None,
                          delta_color="off")
                if dri1 and dri1 > 1:
                    st.success(f"Combo uses **{dri1:.1f}x less** {d1_label}.")
            else:
                st.info(f"{d1_label} alone cannot reach {opt['viability']:.1f}%.")
        with cmp2:
            mono2 = opt.get("mono2_equiv")
            dri2 = opt.get("dri2")
            if mono2 is not None:
                st.metric(f"{d2_label} alone", f"{mono2:.4g}",
                          delta=f"{dri2:.1f}x dose reduction" if dri2 else None,
                          delta_color="off")
                if dri2 and dri2 > 1:
                    st.success(f"Combo uses **{dri2:.1f}x less** {d2_label}.")
            else:
                st.info(f"{d2_label} alone cannot reach {opt['viability']:.1f}%.")

        # Dose bar chart
        bar_names, bar_vals, bar_colors = [], [], []
        if opt.get("mono1_equiv"):
            bar_names += [f"{d1_label}\n(alone)", f"{d1_label}\n(combo)"]
            bar_vals += [opt["mono1_equiv"], opt["d1_opt"]]
            bar_colors += ["#e74c3c", "#27ae60"]
        if opt.get("mono2_equiv"):
            bar_names += [f"{d2_label}\n(alone)", f"{d2_label}\n(combo)"]
            bar_vals += [opt["mono2_equiv"], opt["d2_opt"]]
            bar_colors += ["#e67e22", "#2980b9"]

        if bar_names:
            fig_bar = go.Figure(go.Bar(
                x=bar_names, y=bar_vals, marker_color=bar_colors,
                text=[f"{v:.3g}" for v in bar_vals], textposition="outside"))
            fig_bar.update_layout(
                title="Dose required — monotherapy vs combination",
                yaxis_title="Dose", height=340, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
