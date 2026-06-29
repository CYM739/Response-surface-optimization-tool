# src/views/braid_predict_view.py
"""
BRAID Surface Prediction tab — uses the trained v19 neural network
to predict drug combination response surfaces for ANY drug pair + cell line,
including unseen combinations not in the pre-fitted database.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path
import os


# ── Paths ────────────────────────────────────────────────────────────────────
_SYNERGY_ROOT = Path(os.environ.get(
    "SYNERGY_ROOT", "C:/Max/Github/synergy_docker"))
_DATA_PATH = _SYNERGY_ROOT / "data"


@st.cache_resource(show_spinner="Loading BRAID prediction model (one-time)…")
def _load_engine(checkpoint):
    from logic.braid_predict import BRAIDPredictionEngine
    return BRAIDPredictionEngine(checkpoint=checkpoint)


@st.cache_data(show_spinner=False)
def _load_cell_name_map():
    """Build DepMap_ID → friendly cell line name mapping."""
    si_path = _DATA_PATH / "sample_info.csv"
    if si_path.exists():
        si = pd.read_csv(si_path, usecols=["DepMap_ID", "stripped_cell_line_name"],
                         low_memory=False)
        return dict(zip(si["DepMap_ID"], si["stripped_cell_line_name"]))
    return {}


def render():
    st.header("🧬 BRAID Surface Predictor")
    st.caption(
        "Neural network prediction of drug combination surfaces. "
        "Uses the trained v19 model to predict BRAID parameters (E0, Einf, κ) "
        "for any drug pair + cell line — including novel combinations not in the database."
    )

    # ── 1. Model loading ─────────────────────────────────────────────────────
    with st.sidebar.expander("Predictor Settings", expanded=False):
        checkpoint = st.selectbox(
            "Model checkpoint",
            options=[
                "best_v19_cold_drug.pth",
                "best_v19_random.pth",
                "best_v19_cold_cell.pth",
            ],
            index=0,
            help=(
                "cold_drug: best for predicting new drugs. "
                "cold_cell: best for new cell lines. "
                "random: general-purpose."
            ),
            key="bp_ckpt",
        )

    try:
        engine = _load_engine(checkpoint)
    except Exception as e:
        st.error(
            f"Failed to load prediction model: {e}\n\n"
            "Ensure PyTorch, torch_geometric, and rdkit are installed, "
            "and that synergy_docker data files are accessible."
        )
        return

    drugs = engine.get_available_drugs()
    cells = engine.get_available_cells()
    cell_names = _load_cell_name_map()

    # Build display labels for cells: "CellName (ACH-000XXX)"
    cell_display = {
        cid: f"{cell_names.get(cid, cid)} ({cid})" if cid in cell_names else cid
        for cid in cells
    }
    cell_labels = sorted(cell_display.values())
    label_to_id = {v: k for k, v in cell_display.items()}

    # ── 2. Input controls ────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 2, 2])
    drug1 = col1.selectbox("Drug 1", drugs, key="bp_drug1")
    drug2 = col2.selectbox("Drug 2", drugs, index=min(1, len(drugs) - 1), key="bp_drug2")
    cell_label = col3.selectbox("Cell Line", cell_labels, key="bp_cell")
    cell_id = label_to_id.get(cell_label, cell_label)

    if drug1 == drug2:
        st.warning("Drug 1 and Drug 2 are the same. Select two different drugs.")
        return

    # ── 3. Monotherapy parameters ────────────────────────────────────────────
    mono_lookup = engine.lookup_mono_params(drug1, drug2, cell_id)

    # Track drug/cell selection changes to reset mono params
    _sel_key = f"{drug1}|{drug2}|{cell_id}"
    if st.session_state.get("_bp_sel_key") != _sel_key:
        st.session_state["_bp_sel_key"] = _sel_key
        st.session_state["bp_ec50_1"] = mono_lookup["EC50_1"] if mono_lookup else 1.0
        st.session_state["bp_h1"] = mono_lookup["h1"] if mono_lookup else 1.0
        st.session_state["bp_ec50_2"] = mono_lookup["EC50_2"] if mono_lookup else 1.0
        st.session_state["bp_h2"] = mono_lookup["h2"] if mono_lookup else 1.0

    with st.expander("Monotherapy Parameters (EC50 & Hill slope)", expanded=True):
        if mono_lookup:
            st.success("Auto-filled from the fitted surface database.")
        else:
            st.info(
                "No fitted data found for this combination. "
                "Enter monotherapy parameters manually (from dose-response experiments)."
            )

        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(f"**{drug1}**")
            ec50_1 = st.number_input(
                "EC50₁", min_value=1e-12, format="%.4g", key="bp_ec50_1",
                help="Half-maximal effective concentration for Drug 1",
            )
            h1 = st.number_input(
                "Hill slope h₁", min_value=0.01, format="%.3f", key="bp_h1",
                help="Hill coefficient (steepness) for Drug 1",
            )
        with mc2:
            st.markdown(f"**{drug2}**")
            ec50_2 = st.number_input(
                "EC50₂", min_value=1e-12, format="%.4g", key="bp_ec50_2",
                help="Half-maximal effective concentration for Drug 2",
            )
            h2 = st.number_input(
                "Hill slope h₂", min_value=0.01, format="%.3f", key="bp_h2",
                help="Hill coefficient (steepness) for Drug 2",
            )

    mono_params = {"EC50_1": ec50_1, "h1": h1, "EC50_2": ec50_2, "h2": h2}

    # ── 4. Predict ───────────────────────────────────────────────────────────
    predict_btn = st.button("Predict Surface", type="primary", key="bp_predict")

    if predict_btn:
        with st.spinner("Running inference…"):
            try:
                result = engine.predict_surface(drug1, drug2, cell_id, mono_params)
                st.session_state["bp_result"] = result
            except Exception as e:
                import traceback
                st.error(f"Prediction failed: {e}")
                st.code(traceback.format_exc())
                return

    result = st.session_state.get("bp_result")
    if result is None:
        st.info("Select drugs and cell line, set monotherapy parameters, then click **Predict Surface**.")
        return

    # ── 5. Results: Parameters + Kappa badge ─────────────────────────────────
    d1_label = result["drug1"]
    d2_label = result["drug2"]
    cl_label = cell_names.get(result["cell_line"], result["cell_line"])

    p_col, b_col = st.columns([3, 2])

    with p_col:
        st.subheader("Predicted BRAID Parameters")
        st.table({
            "Parameter": [
                "EC50₁ (input)", "h₁ — Hill slope (input)",
                "EC50₂ (input)", "h₂ — Hill slope (input)",
                "E0 — baseline % (predicted)", "Einf — max effect % (predicted)",
                "κ — kappa (predicted)",
            ],
            "Value": [
                f"{result['EC50_1']:.4g}",
                f"{result['h1']:.3f}",
                f"{result['EC50_2']:.4g}",
                f"{result['h2']:.3f}",
                f"{result['E0']:.1f}",
                f"{result['Einf']:.1f}",
                f"{result['kappa']:.3f}",
            ],
        })
        st.caption(
            f"Model: `{checkpoint}` · Cell: **{cl_label}** ({result['cell_line']})"
        )

    with b_col:
        label = result["kappa_label"]
        color = result["kappa_color"]
        st.markdown(
            f"""
            <div style="background:{color}22; border:2px solid {color};
                        border-radius:12px; padding:24px; text-align:center; margin-top:16px">
                <div style="font-size:1.8em; font-weight:bold">{label}</div>
                <div style="font-size:1.5em; font-weight:bold; margin-top:6px">
                    κ = {result['kappa']:.3f}
                </div>
                <div style="font-size:0.82em; color:#666; margin-top:10px">
                    κ &gt; 1 → synergistic<br>
                    −1 ≤ κ ≤ 1 → additive<br>
                    κ &lt; −1 → antagonistic
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── 6. Visualisation tabs ────────────────────────────────────────────────
    d1_arr = result["d1_arr"]
    d2_arr = result["d2_arr"]
    V_grid = result["V_grid"]

    vt1, vt2, vt3 = st.tabs(["3D Surface", "2D Heatmap", "Dose-Response Slices"])

    with vt1:
        fig3d = go.Figure(data=[go.Surface(
            x=d1_arr, y=d2_arr, z=V_grid,
            colorscale="RdBu", reversescale=True,
            colorbar=dict(title="Viability %", thickness=14),
            hovertemplate=(
                f"{d1_label}: %{{x:.3g}}<br>"
                f"{d2_label}: %{{y:.3g}}<br>"
                "Viability: %{z:.1f}%<extra></extra>"
            ),
        )])
        fig3d.update_layout(
            scene=dict(
                xaxis_title=f"{d1_label} dose",
                yaxis_title=f"{d2_label} dose",
                zaxis_title="Viability (%)",
                zaxis=dict(range=[
                    max(0, float(V_grid.min()) - 5),
                    min(110, float(V_grid.max()) + 5),
                ]),
            ),
            title=f"Predicted: {d1_label} + {d2_label} in {cl_label}",
            height=540,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig3d, use_container_width=True)

    with vt2:
        fig2d = go.Figure(data=go.Heatmap(
            x=d1_arr, y=d2_arr, z=V_grid,
            colorscale="RdBu", reversescale=True,
            colorbar=dict(title="Viability %"),
            hovertemplate=(
                f"{d1_label}: %{{x:.3g}}<br>"
                f"{d2_label}: %{{y:.3g}}<br>"
                "Viability: %{z:.1f}%<extra></extra>"
            ),
        ))
        fig2d.update_layout(
            xaxis_title=f"{d1_label} dose",
            yaxis_title=f"{d2_label} dose",
            title=f"Viability Heatmap — {d1_label} + {d2_label} in {cl_label}",
            height=480,
        )
        st.plotly_chart(fig2d, use_container_width=True)

    with vt3:
        _render_dose_slices(result, d1_label, d2_label, cl_label)

    # ── 7. Comparison with lookup (if available) ─────────────────────────────
    _render_comparison(engine, result, d1_label, d2_label, cl_label)


def _render_dose_slices(result, d1_label, d2_label, cl_label):
    """1D dose-response curves at fixed doses of the other drug."""
    d1_arr = result["d1_arr"]
    d2_arr = result["d2_arr"]
    V_grid = result["V_grid"]  # shape [n_d2, n_d1]
    n = len(d1_arr)

    # Pick a few slices through the surface
    slice_indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]

    sc1, sc2 = st.columns(2)

    with sc1:
        fig = go.Figure()
        for i in slice_indices:
            d2_val = d2_arr[i]
            fig.add_trace(go.Scatter(
                x=d1_arr, y=V_grid[i, :],
                mode="lines",
                name=f"{d2_label}={d2_val:.3g}",
            ))
        fig.update_layout(
            title=f"{d1_label} dose-response at fixed {d2_label}",
            xaxis_title=f"{d1_label} dose",
            yaxis_title="Viability (%)",
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with sc2:
        fig = go.Figure()
        for i in slice_indices:
            d1_val = d1_arr[i]
            fig.add_trace(go.Scatter(
                x=d2_arr, y=V_grid[:, i],
                mode="lines",
                name=f"{d1_label}={d1_val:.3g}",
            ))
        fig.update_layout(
            title=f"{d2_label} dose-response at fixed {d1_label}",
            xaxis_title=f"{d2_label} dose",
            yaxis_title="Viability (%)",
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_comparison(engine, pred_result, d1_label, d2_label, cl_label):
    """If the combination exists in the lookup DB, show predicted vs fitted."""
    from logic.braid_lookup import load_braid_index, query_surface, reconstruct_surface

    try:
        braid_path = str(_SYNERGY_ROOT / "braid_surface_prediction" / "data")
        index_df = load_braid_index(braid_path)
        lookup = query_surface(index_df, pred_result["drug1"],
                               pred_result["drug2"], cl_label)
    except Exception:
        lookup = None

    if lookup is None:
        return

    st.divider()
    st.subheader("Predicted vs Fitted (database)")
    st.caption(
        "This combination exists in the pre-fitted database. "
        "Compare the neural network prediction against the curve-fitted ground truth."
    )

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("κ predicted", f"{pred_result['kappa']:.3f}")
    mc2.metric("κ fitted", f"{lookup['kappa']:.3f}")
    mc3.metric("Δκ", f"{pred_result['kappa'] - lookup['kappa']:.3f}")

    # Side-by-side surfaces
    d1_fit, d2_fit, V_fit = reconstruct_surface(lookup, n_points=60)

    sc1, sc2 = st.columns(2)

    with sc1:
        fig = go.Figure(data=[go.Surface(
            x=pred_result["d1_arr"], y=pred_result["d2_arr"], z=pred_result["V_grid"],
            colorscale="RdBu", reversescale=True,
            colorbar=dict(title="V%", thickness=10),
        )])
        fig.update_layout(
            scene=dict(
                xaxis_title=f"{d1_label}",
                yaxis_title=f"{d2_label}",
                zaxis_title="Viability (%)",
            ),
            title="Predicted (v19 model)",
            height=420, margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with sc2:
        fig = go.Figure(data=[go.Surface(
            x=d1_fit, y=d2_fit, z=V_fit,
            colorscale="RdBu", reversescale=True,
            colorbar=dict(title="V%", thickness=10),
        )])
        fig.update_layout(
            scene=dict(
                xaxis_title=f"{d1_label}",
                yaxis_title=f"{d2_label}",
                zaxis_title="Viability (%)",
            ),
            title="Fitted (ground truth)",
            height=420, margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
