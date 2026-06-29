# src/views/braid_lookup_view.py
"""
BRAID Surface Lookup tab — pure Streamlit view.
Searches the pre-fitted BRAID surface database (216,846 drug-combination surfaces)
and renders interactive 3D/2D plots + parameter table + kappa badge.
No analysis_done requirement — works standalone.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

from logic.braid_lookup import (
    load_braid_index,
    get_available_options,
    query_surface,
    reconstruct_surface,
    kappa_label,
    load_raw_checkerboard,
    optimize_braid_dose,
)

# ── Paths (same as logic layer defaults, override via env) ─────────────────────
import os
_BRAID_PATH = Path(os.environ.get("BRAID_PATH",
    "C:/Max/Github/synergy_docker/braid_surface_prediction/data"))
_DATA_PATH  = Path(os.environ.get("DATA_PATH",
    "C:/Max/Github/synergy_docker/data"))


@st.cache_data(show_spinner="Loading BRAID surface index…")
def _cached_load_index(braid_path_str: str):
    return load_braid_index(braid_path_str)


def render():
    st.header("🔬 BRAID Surface Lookup")
    st.caption(
        "Real experimental data — 216,846 fitted drug combination surfaces "
        "from DrugComb (ONEIL + NCI-ALMANAC). No analysis required."
    )

    # ── 1. Load index (cached once per session) ────────────────────────────────
    try:
        index_df = _cached_load_index(str(_BRAID_PATH))
    except FileNotFoundError as e:
        st.error(
            f"BRAID index not found: {e}\n\n"
            "Set the `BRAID_PATH` environment variable to the folder containing "
            "`braid_labels_qc.parquet`."
        )
        return

    drugs, cell_lines = get_available_options(index_df)

    # ── 2. Search controls ─────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    drug1     = col1.selectbox("Drug 1",     drugs,      key="bl_drug1")
    drug2     = col2.selectbox("Drug 2",     drugs,      key="bl_drug2")
    cell_line = col3.selectbox("Cell Line",  cell_lines, key="bl_cell")
    search    = col4.button("Search", width="stretch", key="bl_search")

    # ── 3. Query ───────────────────────────────────────────────────────────────
    if search:
        result = query_surface(index_df, drug1, drug2, cell_line)
        st.session_state["braid_lookup_result"] = result
        st.session_state["braid_lookup_query"]  = (drug1, drug2, cell_line)

    result = st.session_state.get("braid_lookup_result")

    if result is None and not search:
        st.info("Select Drug 1, Drug 2, and Cell Line, then click **Search**.")
        return

    if result is None:
        q = st.session_state.get("braid_lookup_query", (drug1, drug2, cell_line))
        st.warning(
            f"No surface found for **{q[0]}** + **{q[1]}** in **{q[2]}**. "
            "Try swapping drug order or selecting a different cell line."
        )
        return

    # Restore display names from the stored query
    q = st.session_state.get("braid_lookup_query", (drug1, drug2, cell_line))
    d1_label, d2_label, cl_label = q

    # ── 4. Parameter panel + kappa badge ──────────────────────────────────────
    p_col, b_col = st.columns([3, 2])

    with p_col:
        st.subheader("BRAID Parameters")
        st.table({
            "Parameter": [
                "EC50_1", "h1 (Hill slope)",
                "EC50_2", "h2 (Hill slope)",
                "E0 (baseline %)", "Einf (max effect %)",
                "κ (kappa)", "R²", "RMSE",
            ],
            "Value": [
                f"{result['EC50_1']:.4g}",
                f"{result['h1']:.3f}",
                f"{result['EC50_2']:.4g}",
                f"{result['h2']:.3f}",
                f"{result['E0']:.1f}",
                f"{result['Einf']:.1f}",
                f"{result['kappa']:.3f}",
                f"{result['r2_braid']:.3f}",
                f"{result['rmse_braid']:.2f}",
            ],
        })
        st.caption(
            f"Source: **{result['source']}** · BlockID: `{result['BlockID']}` · "
            f"Drug order stored as: **{result['drug1']}** + **{result['drug2']}**"
        )

    with b_col:
        label, color = kappa_label(result["kappa"])
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

    # ── 5. Reconstruct surface once ────────────────────────────────────────────
    d1_arr, d2_arr, V_grid = reconstruct_surface(result, n_points=60)

    # ── 6. Visualisation tabs ──────────────────────────────────────────────────
    vt1, vt2, vt3 = st.tabs(["3D Surface", "2D Heatmap", "Raw Data"])

    with vt1:
        fig3d = go.Figure(data=[go.Surface(
            x=d1_arr,
            y=d2_arr,
            z=V_grid,
            colorscale="RdBu",
            reversescale=True,
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
            title=f"{d1_label} + {d2_label} in {cl_label}",
            height=540,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig3d, width="stretch")

    with vt2:
        fig2d = go.Figure(data=go.Heatmap(
            x=d1_arr,
            y=d2_arr,
            z=V_grid,
            colorscale="RdBu",
            reversescale=True,
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
        st.plotly_chart(fig2d, width="stretch")

    with vt3:
        raw = load_raw_checkerboard(result["BlockID"], str(_DATA_PATH))
        if raw is not None and not raw.empty:
            st.caption(
                f"Raw measured checkerboard data for BlockID `{result['BlockID']}` "
                f"({len(raw)} data points)"
            )

            # Side-by-side: 3D scatter overlay + table
            sc_col, tbl_col = st.columns([3, 2])

            with sc_col:
                fig_raw = go.Figure()
                # Fitted surface (transparent)
                fig_raw.add_trace(go.Surface(
                    x=d1_arr, y=d2_arr, z=V_grid,
                    colorscale="RdBu", reversescale=True,
                    opacity=0.5,
                    showscale=False,
                    name="Fitted surface",
                ))
                # Raw data scatter
                fig_raw.add_trace(go.Scatter3d(
                    x=raw["ConcRow"].values,
                    y=raw["ConcCol"].values,
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
                    height=480,
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                st.plotly_chart(fig_raw, width="stretch")

            with tbl_col:
                st.dataframe(
                    raw[["ConcRow", "ConcCol", "Response"]].rename(columns={
                        "ConcRow": f"{d1_label} dose",
                        "ConcCol": f"{d2_label} dose",
                        "Response": "Viability %",
                    }).round(3),
                    width="stretch",
                    height=430,
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

    # ── 7. Dose Optimisation Panel ─────────────────────────────────────────────
    st.divider()
    with st.expander("🎯 Dose Optimisation", expanded=False):
        st.caption(
            "Find the optimal drug doses directly from the BRAID surface using scipy. "
            "No model training needed — the closed-form surface is optimised analytically."
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
                key="bl_opt_objective",
            )

        with oc2:
            d1_max_mult = st.slider(
                f"Max dose ({d1_label}) — × EC50",
                min_value=1.0, max_value=10.0, value=4.0, step=0.5,
                key="bl_d1_mult",
            )
            d2_max_mult = st.slider(
                f"Max dose ({d2_label}) — × EC50",
                min_value=1.0, max_value=10.0, value=4.0, step=0.5,
                key="bl_d2_mult",
            )

        if objective == "min_dose":
            v_min_possible = float(result["Einf"])
            v_baseline     = float(result["E0"])
            target_default = round(max(v_min_possible + 5, (v_min_possible + v_baseline) * 0.5), 1)
            target_v = st.slider(
                "Target viability (%)",
                min_value=float(max(0, v_min_possible + 1)),
                max_value=float(v_baseline),
                value=float(min(target_default, v_baseline - 1)),
                step=1.0,
                key="bl_target_v",
                help=f"Einf (floor) = {v_min_possible:.1f}%  |  E0 (baseline) = {v_baseline:.1f}%",
            )

            w_col1, w_col2 = st.columns(2)
            w1 = w_col1.number_input(f"Dose weight — {d1_label}", value=1.0, min_value=0.01,
                                     step=0.1, key="bl_w1",
                                     help="Higher weight = optimizer avoids this drug more")
            w2 = w_col2.number_input(f"Dose weight — {d2_label}", value=1.0, min_value=0.01,
                                     step=0.1, key="bl_w2")
        else:
            target_v = None
            w1, w2   = 1.0, 1.0

        run_opt = st.button("Run Optimisation", key="bl_run_opt")

        if run_opt:
            with st.spinner("Optimising…"):
                opt = optimize_braid_dose(
                    result,
                    objective=objective,
                    target_viability=target_v if target_v is not None else 50.0,
                    d1_max_mult=d1_max_mult,
                    d2_max_mult=d2_max_mult,
                    dose_weight=(w1, w2),
                )
            st.session_state["braid_opt_result"] = opt

        opt = st.session_state.get("braid_opt_result")
        if opt is None:
            st.stop()

        if not opt.get("feasible", True):
            st.warning(opt.get("message", "Optimisation failed."))
            st.stop()

        # ── Results ────────────────────────────────────────────────────────────
        st.subheader("Optimal Doses")
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{d1_label} dose", f"{opt['d1_opt']:.4g}")
        m2.metric(f"{d2_label} dose", f"{opt['d2_opt']:.4g}")
        m3.metric("Predicted viability", f"{opt['viability']:.1f}%")

        # Monotherapy comparison
        st.subheader("Monotherapy Comparison")
        st.caption(
            "How much of each drug would be needed alone to achieve the same viability?"
        )

        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            mono1 = opt.get("mono1_equiv")
            dri1  = opt.get("dri1")
            if mono1 is not None:
                st.metric(
                    f"{d1_label} alone (equiv.)",
                    f"{mono1:.4g}",
                    delta=f"{dri1:.1f}× dose reduction" if dri1 else None,
                    delta_color="off",
                )
                if dri1 and dri1 > 1:
                    st.success(f"Combination uses **{dri1:.1f}× less** {d1_label} than monotherapy.")
                elif dri1 and dri1 < 1:
                    st.warning(f"Combination requires **{1/dri1:.1f}× more** {d1_label} than monotherapy.")
            else:
                st.info(f"{d1_label} monotherapy cannot reach {opt['viability']:.1f}% viability alone (below Einf floor).")

        with cmp_col2:
            mono2 = opt.get("mono2_equiv")
            dri2  = opt.get("dri2")
            if mono2 is not None:
                st.metric(
                    f"{d2_label} alone (equiv.)",
                    f"{mono2:.4g}",
                    delta=f"{dri2:.1f}× dose reduction" if dri2 else None,
                    delta_color="off",
                )
                if dri2 and dri2 > 1:
                    st.success(f"Combination uses **{dri2:.1f}× less** {d2_label} than monotherapy.")
                elif dri2 and dri2 < 1:
                    st.warning(f"Combination requires **{1/dri2:.1f}× more** {d2_label} than monotherapy.")
            else:
                st.info(f"{d2_label} monotherapy cannot reach {opt['viability']:.1f}% viability alone (below Einf floor).")

        # Dose bar chart
        bar_names, bar_vals, bar_colors = [], [], []
        if opt.get("mono1_equiv"):
            bar_names += [f"{d1_label}\n(alone)", f"{d1_label}\n(combo)"]
            bar_vals  += [opt["mono1_equiv"], opt["d1_opt"]]
            bar_colors += ["#e74c3c", "#27ae60"]
        if opt.get("mono2_equiv"):
            bar_names += [f"{d2_label}\n(alone)", f"{d2_label}\n(combo)"]
            bar_vals  += [opt["mono2_equiv"], opt["d2_opt"]]
            bar_colors += ["#e67e22", "#2980b9"]

        if bar_names:
            fig_bar = go.Figure(go.Bar(
                x=bar_names, y=bar_vals,
                marker_color=bar_colors,
                text=[f"{v:.3g}" for v in bar_vals],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title="Dose required — monotherapy vs combination",
                yaxis_title="Dose",
                height=340,
                showlegend=False,
            )
            st.plotly_chart(fig_bar, width="stretch")
