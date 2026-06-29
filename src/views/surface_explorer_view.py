# src/views/surface_explorer_view.py
"""
Surface Explorer — unified tab for visualizing and optimizing ANY Surface.

Works with:
  - ModelSurface (OLS / SVR / RF from user CSV)
  - BRAIDSurface (lookup or NN prediction)

This is a standalone test tab — does not modify any existing views.
"""

import os
import json

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import numpy as np
import pandas as pd


# Explorer mode options. `se_view_mode_saved` (a plain session_state key, not
# a widget key) mirrors the choice so it survives reruns where this tab is not
# re-rendered — e.g. a Project Library button that calls st.rerun() before
# tab 9 runs, which otherwise garbage-collects the radio's widget state and
# silently drops the user back to Classic mode.
_VIEW_OPTIONS = ["Classic (server-side)", "⚡ Live (client-side — instant sliders)"]


def render():
    st.header("Surface Explorer")
    st.caption(
        "Unified surface visualization, optimization, and comparison. "
        "Works with fitted models (from Project Library) and BRAID surfaces alike."
    )

    saved = st.session_state.get("se_view_mode_saved", _VIEW_OPTIONS[0])
    if saved not in _VIEW_OPTIONS:
        saved = _VIEW_OPTIONS[0]
    view_mode = st.radio(
        "Explorer mode", _VIEW_OPTIONS,
        index=_VIEW_OPTIONS.index(saved),
        horizontal=True, key="se_view_mode",
        help="Live mode fits a quadratic surface and renders it entirely in "
             "your browser — slider/axis changes update with no Streamlit rerun.",
    )
    st.session_state["se_view_mode_saved"] = view_mode

    if view_mode == _VIEW_OPTIONS[1]:
        _render_client_side()
        return

    # ── 1. Collect available surfaces ────────────────────────────────────────
    surfaces = _collect_surfaces()

    if not surfaces:
        st.info(
            "No surfaces available. To use this tab:\n"
            "- Load a project and run an analysis in the **Project Library** tab, or\n"
            "- Search/predict a BRAID surface in the **BRAID Surface** tab."
        )
        return

    # ── 2. Surface selector ──────────────────────────────────────────────────
    surf_names = list(surfaces.keys())

    mode = st.radio(
        "Mode", ["Single Surface", "Compare Two Surfaces"],
        horizontal=True, key="se_mode")

    if mode == "Single Surface":
        sel = st.selectbox("Surface", surf_names, key="se_surf1")
        surf = surfaces[sel]
        _render_single(surf, sel)
    else:
        c1, c2 = st.columns(2)
        sel1 = c1.selectbox("Surface A", surf_names, key="se_surfA")
        sel2 = c2.selectbox("Surface B", surf_names,
                            index=min(1, len(surf_names) - 1), key="se_surfB")
        if sel1 == sel2:
            st.warning("Select two different surfaces to compare.")
            return
        _render_comparison(surfaces[sel1], sel1, surfaces[sel2], sel2)


# ═════════════════════════════════════════════════════════════════════════════
#  Client-side (live) explorer — embeds surface_explorer_client.html
# ═════════════════════════════════════════════════════════════════════════════

_CLIENT_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "surface_explorer_client.html")


def _session_injection(session=None):
    """Build the data blob to hand the client-side explorer from the loaded
    project, or None if no usable analyzed data is in session.

    `session` defaults to st.session_state; it is parameterised so the
    builder can be unit-tested with a plain dict.
    """
    session = st.session_state if session is None else session
    df = session.get("exp_df")
    indep = session.get("independent_vars") or []
    dep = session.get("dependent_vars") or []
    if df is None or not indep or not dep:
        return None
    cols = [c for c in (list(indep) + list(dep)) if c in df.columns]
    if len(cols) < 3:
        return None
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 3:
        return None
    descs = session.get("variable_descriptions") or {}
    labels = {c: descs[c] for c in sub.columns
              if c in descs and descs[c] and descs[c] != c}
    return {
        "headers": list(sub.columns),
        "rows": sub.values.astype(float).tolist(),
        "response": dep[0] if dep[0] in sub.columns else sub.columns[-1],
        "labels": labels,
    }


def _render_client_side():
    """Render the browser-side surface explorer inside an iframe component."""
    try:
        with open(_CLIENT_HTML, encoding="utf-8") as fh:
            html = fh.read()
    except OSError as exc:
        st.error(f"Could not load the client-side explorer file: {exc}")
        return

    # Inline the vendored Plotly so the embedded iframe has zero external
    # (CDN) dependency — matters on locked-down coworker networks.
    _tag = '<script src="plotly-2.32.0.min.js"></script>'
    _plotly = os.path.join(os.path.dirname(_CLIENT_HTML), "plotly-2.32.0.min.js")
    if os.path.exists(_plotly):
        with open(_plotly, encoding="utf-8") as fh:
            html = html.replace(_tag, "<script>" + fh.read() + "</script>", 1)
    else:
        html = html.replace(
            _tag, '<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>', 1)

    inj = _session_injection()
    if inj is not None:
        blob = json.dumps(inj)
        html = html.replace("var INJECTED = null;",
                             f"var INJECTED = {blob};", 1)
        st.caption("Pre-loaded with your analyzed project data. Slider and axis "
                   "changes recompute in your browser — no Streamlit rerun.")
    else:
        st.caption("No analyzed project detected — upload a drug-combination CSV "
                   "in the explorer below. All interaction runs client-side.")

    components.html(html, height=900, scrolling=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Surface collection
# ═════════════════════════════════════════════════════════════════════════════

def _collect_surfaces():
    """Gather all available surfaces from session state."""
    from logic.surface import surfaces_from_session, BRAIDSurface

    result = {}

    # 1. Model surfaces from analysis
    model_surfs = surfaces_from_session(st.session_state)
    for name, surf in model_surfs.items():
        result[f"Model: {name}"] = surf

    # 2. BRAID lookup surface (from BRAID Surface tab)
    braid_lookup = st.session_state.get("bs_lookup")
    braid_query = st.session_state.get("bs_query")
    if braid_lookup is not None and braid_query is not None:
        from logic.surface import surface_from_braid_lookup
        d1, d2, cl = braid_query
        label = f"BRAID Lookup: {d1} + {d2} in {cl}"
        result[label] = surface_from_braid_lookup(braid_lookup)

    # 3. BRAID prediction surface (from BRAID Surface tab)
    braid_pred = st.session_state.get("bs_prediction")
    if braid_pred is not None:
        from logic.surface import surface_from_braid_prediction
        label = f"BRAID Predicted: {braid_pred['drug1']} + {braid_pred['drug2']}"
        result[label] = surface_from_braid_prediction(braid_pred)

    # 4. BRAID comparison prediction (from NN comparison in BRAID Surface tab)
    braid_cmp = st.session_state.get("bs_comparison")
    if braid_cmp is not None:
        from logic.surface import surface_from_braid_prediction
        label = f"BRAID NN Compare: {braid_cmp['drug1']} + {braid_cmp['drug2']}"
        result[label] = surface_from_braid_prediction(braid_cmp, checkpoint_name="comparison")

    return result


# ═════════════════════════════════════════════════════════════════════════════
#  Single surface view
# ═════════════════════════════════════════════════════════════════════════════

def _render_single(surf, name):
    """Full exploration of a single surface."""

    # ── Info panel ────────────────────────────────────────────────────────────
    with st.expander("Surface Info", expanded=False):
        ic1, ic2 = st.columns([3, 2])
        with ic1:
            st.table({"Parameter": [p[0] for p in surf.params_table],
                       "Value": [p[1] for p in surf.params_table]})
        with ic2:
            st.code(surf.summary(), language=None)

    # ── Variable selection ───────────────────────────────────────────────────
    vars_ = surf.var_names
    if len(vars_) < 2:
        st.warning("Surface needs at least 2 variables for 3D visualization.")
        return

    vc1, vc2 = st.columns(2)
    x_var = vc1.selectbox("X axis", vars_, index=0, key="se_xvar")
    y_var = vc2.selectbox("Y axis", vars_, index=min(1, len(vars_) - 1), key="se_yvar")

    if x_var == y_var:
        st.warning("X and Y axes must be different variables.")
        return

    # Fixed variables (for surfaces with >2 vars)
    fixed = {}
    other_vars = [v for v in vars_ if v not in (x_var, y_var)]
    if other_vars:
        with st.expander(f"Fixed variables ({len(other_vars)})", expanded=False):
            for v in other_vars:
                lo, hi = surf.bounds[v]
                mid = (lo + hi) / 2
                fixed[v] = st.slider(v, lo, hi, mid, key=f"se_fix_{v}")

    # ── Generate grid ────────────────────────────────────────────────────────
    arr1, arr2, Z = surf.predict_grid(x_var, y_var, n_points=60, fixed=fixed)

    # ── Visualization tabs ───────────────────────────────────────────────────
    vt1, vt2, vt3 = st.tabs(["3D Surface", "2D Heatmap", "Dose-Response Slices"])

    with vt1:
        _plot_3d(arr1, arr2, Z, x_var, y_var, surf.response_name,
                 title=name, key_suffix="single")

    with vt2:
        _plot_heatmap(arr1, arr2, Z, x_var, y_var, surf.response_name)

    with vt3:
        _plot_slices(arr1, arr2, Z, x_var, y_var, surf.response_name)

    # ── Optimization ─────────────────────────────────────────────────────────
    st.divider()
    _render_optimization(surf, x_var, y_var, fixed, arr1, arr2, Z)

    # ── Synergy (only for 2-var surfaces) ────────────────────────────────────
    if len(vars_) == 2:
        st.divider()
        _render_synergy(surf, x_var, y_var, fixed)


# ═════════════════════════════════════════════════════════════════════════════
#  Comparison view
# ═════════════════════════════════════════════════════════════════════════════

def _render_comparison(surf_a, name_a, surf_b, name_b):
    """Side-by-side comparison of two surfaces."""

    # Find common variables
    common_vars = [v for v in surf_a.var_names if v in surf_b.var_names]

    if len(common_vars) < 2:
        st.warning(
            f"Surfaces share {len(common_vars)} variable(s) in common. "
            "Need at least 2 for comparison."
        )
        st.write(f"**{name_a}** vars: {surf_a.var_names}")
        st.write(f"**{name_b}** vars: {surf_b.var_names}")
        return

    vc1, vc2 = st.columns(2)
    x_var = vc1.selectbox("X axis", common_vars, index=0, key="se_cmp_x")
    y_var = vc2.selectbox("Y axis", common_vars,
                          index=min(1, len(common_vars) - 1), key="se_cmp_y")

    if x_var == y_var:
        st.warning("X and Y axes must be different.")
        return

    # Use union of bounds
    x_lo = min(surf_a.bounds[x_var][0], surf_b.bounds[x_var][0])
    x_hi = max(surf_a.bounds[x_var][1], surf_b.bounds[x_var][1])
    y_lo = min(surf_a.bounds[y_var][0], surf_b.bounds[y_var][0])
    y_hi = max(surf_a.bounds[y_var][1], surf_b.bounds[y_var][1])

    n = 60
    arr1 = np.linspace(x_lo, x_hi, n)
    arr2 = np.linspace(y_lo, y_hi, n)
    G1, G2 = np.meshgrid(arr1, arr2)
    base_df = pd.DataFrame({x_var: G1.ravel(), y_var: G2.ravel()})

    # Fill other vars with midpoint for each surface
    def _predict_on_grid(surf, df):
        df_copy = df.copy()
        for v in surf.var_names:
            if v not in (x_var, y_var):
                lo, hi = surf.bounds[v]
                df_copy[v] = (lo + hi) / 2
        return surf.predict(df_copy).reshape(n, n)

    Z_a = _predict_on_grid(surf_a, base_df)
    Z_b = _predict_on_grid(surf_b, base_df)
    Z_diff = Z_a - Z_b

    # ── Info comparison ──────────────────────────────────────────────────────
    with st.expander("Surface Info", expanded=False):
        ic1, ic2 = st.columns(2)
        with ic1:
            st.markdown(f"**{name_a}**")
            st.code(surf_a.summary(), language=None)
        with ic2:
            st.markdown(f"**{name_b}**")
            st.code(surf_b.summary(), language=None)

    # ── Delta metrics (for BRAID surfaces) ───────────────────────────────────
    from logic.surface import BRAIDSurface
    if isinstance(surf_a, BRAIDSurface) and isinstance(surf_b, BRAIDSurface):
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric(f"kappa A", f"{surf_a.kappa:.3f}")
        mc2.metric(f"kappa B", f"{surf_b.kappa:.3f}")
        mc3.metric("Delta kappa", f"{surf_a.kappa - surf_b.kappa:.3f}")
        mc4.metric("Mean |diff|", f"{np.abs(Z_diff).mean():.2f}%")

    # ── Side-by-side 3D ──────────────────────────────────────────────────────
    sc1, sc2 = st.columns(2)
    with sc1:
        _plot_3d(arr1, arr2, Z_a, x_var, y_var, surf_a.response_name,
                 title=name_a, height=420, key_suffix="cmpA")
    with sc2:
        _plot_3d(arr1, arr2, Z_b, x_var, y_var, surf_b.response_name,
                 title=name_b, height=420, key_suffix="cmpB")

    # ── Difference surface ───────────────────────────────────────────────────
    st.subheader("Difference (A - B)")
    _plot_heatmap(arr1, arr2, Z_diff, x_var, y_var, "Difference",
                  colorscale="RdBu", zmid=0)


# ═════════════════════════════════════════════════════════════════════════════
#  Optimization panel
# ═════════════════════════════════════════════════════════════════════════════

def _render_optimization(surf, x_var, y_var, fixed, arr1, arr2, Z):
    with st.expander("Optimization", expanded=False):
        oc1, oc2 = st.columns([1, 1])
        with oc1:
            objective = st.radio(
                "Objective",
                ["minimize", "maximize", "target"],
                format_func=lambda x: {
                    "minimize": "Minimize response",
                    "maximize": "Maximize response",
                    "target": "Hit target value",
                }[x],
                key="se_opt_obj",
            )
        with oc2:
            target_val = None
            if objective == "target":
                z_min, z_max = float(Z.min()), float(Z.max())
                target_val = st.number_input(
                    "Target value", value=round((z_min + z_max) / 2, 2),
                    key="se_opt_target")

        run_opt = st.button("Optimize", key="se_opt_run")

        if run_opt:
            with st.spinner("Optimizing..."):
                result = surf.optimize(
                    objective=objective,
                    target_value=target_val,
                    fixed=fixed,
                )
            st.session_state["se_opt_result"] = result

        opt = st.session_state.get("se_opt_result")
        if opt is None:
            return

        if not opt.get("success", False):
            st.warning("Optimization did not converge.")
            return

        # Display results
        st.subheader("Optimal Point")
        cols = st.columns(len(surf.var_names) + 1)
        for i, v in enumerate(surf.var_names):
            cols[i].metric(v, f"{opt[v]:.4g}")
        cols[-1].metric(surf.response_name, f"{opt['response']:.4g}")


# ═════════════════════════════════════════════════════════════════════════════
#  Synergy panel
# ═════════════════════════════════════════════════════════════════════════════

def _render_synergy(surf, var1, var2, fixed):
    with st.expander("Synergy Analysis", expanded=False):
        method = st.radio("Method", ["hsa", "bliss"],
                          format_func=lambda x: {"hsa": "Highest Single Agent (HSA)",
                                                  "bliss": "Bliss Independence"}[x],
                          horizontal=True, key="se_syn_method")

        run_syn = st.button("Compute Synergy", key="se_syn_run")

        if run_syn:
            with st.spinner("Computing synergy grid..."):
                a1, a2, Z_resp, Z_syn = surf.synergy_grid(
                    var1, var2, n_points=50, fixed=fixed, method=method)
                st.session_state["se_synergy"] = (a1, a2, Z_resp, Z_syn)

        syn_data = st.session_state.get("se_synergy")
        if syn_data is None:
            return

        a1, a2, Z_resp, Z_syn = syn_data

        sc1, sc2 = st.columns(2)
        with sc1:
            st.metric("Mean synergy score", f"{Z_syn.mean():.2f}")
            st.metric("Min (most synergistic)", f"{Z_syn.min():.2f}")
        with sc2:
            st.metric("Max (most antagonistic)", f"{Z_syn.max():.2f}")
            pct_syn = (Z_syn < 0).mean() * 100
            st.metric("% grid synergistic", f"{pct_syn:.1f}%")

        # Synergy heatmap
        fig = go.Figure(data=go.Heatmap(
            x=a1, y=a2, z=Z_syn,
            colorscale=[[0, "#27ae60"], [0.5, "#cccccc"], [1, "#e74c3c"]],
            zmid=0,
            colorbar=dict(title="Synergy"),
            hovertemplate=(
                f"{var1}: %{{x:.3g}}<br>"
                f"{var2}: %{{y:.3g}}<br>"
                "Synergy: %{z:.2f}<extra></extra>"
            ),
        ))
        fig.update_layout(
            xaxis_title=f"{var1} dose", yaxis_title=f"{var2} dose",
            title=f"Synergy ({method.upper()}) — negative = synergistic",
            height=450)
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Shared plot helpers
# ═════════════════════════════════════════════════════════════════════════════

def _plot_3d(arr1, arr2, Z, x_label, y_label, z_label,
             title=None, height=540, key_suffix=""):
    use_log = st.checkbox("Log-scale dose axes", value=False,
                          key=f"se_log3d_{key_suffix}" if key_suffix else None)

    if use_log:
        eps1 = arr1[arr1 > 0].min() / 10 if (arr1 > 0).any() else 1e-6
        eps2 = arr2[arr2 > 0].min() / 10 if (arr2 > 0).any() else 1e-6
        x_plot = np.log10(np.where(arr1 > 0, arr1, eps1))
        y_plot = np.log10(np.where(arr2 > 0, arr2, eps2))
        x_title = f"{x_label} (log10)"
        y_title = f"{y_label} (log10)"
    else:
        x_plot, y_plot = arr1, arr2
        x_title, y_title = x_label, y_label

    fig = go.Figure(data=[go.Surface(
        x=x_plot, y=y_plot, z=Z,
        colorscale="RdBu", reversescale=True,
        colorbar=dict(title=z_label, thickness=14),
        customdata=np.dstack(np.meshgrid(arr1, arr2)),
        hovertemplate=(
            f"{x_label}: %{{customdata[0]:.3g}}<br>"
            f"{y_label}: %{{customdata[1]:.3g}}<br>"
            f"{z_label}: %{{z:.2f}}<extra></extra>"
        ),
    )])
    fig.update_layout(
        scene=dict(
            xaxis_title=x_title, yaxis_title=y_title, zaxis_title=z_label,
            zaxis=dict(range=[
                max(0, float(Z.min()) - 5), min(110, float(Z.max()) + 5),
            ]),
        ),
        title=title, height=height,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_heatmap(arr1, arr2, Z, x_label, y_label, z_label,
                  colorscale="RdBu", zmid=None):
    fig = go.Figure(data=go.Heatmap(
        x=arr1, y=arr2, z=Z,
        colorscale=colorscale,
        reversescale=(colorscale == "RdBu"),
        zmid=zmid,
        colorbar=dict(title=z_label),
        hovertemplate=(
            f"{x_label}: %{{x:.3g}}<br>"
            f"{y_label}: %{{y:.3g}}<br>"
            f"{z_label}: %{{z:.2f}}<extra></extra>"
        ),
    ))
    fig.update_layout(
        xaxis_title=x_label, yaxis_title=y_label,
        title=f"{z_label} — {x_label} vs {y_label}",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_slices(arr1, arr2, Z, x_label, y_label, z_label):
    n = len(arr1)
    slices = [0, n // 4, n // 2, 3 * n // 4, n - 1]

    sc1, sc2 = st.columns(2)
    with sc1:
        fig = go.Figure()
        for i in slices:
            fig.add_trace(go.Scatter(
                x=arr1, y=Z[i, :], mode="lines",
                name=f"{y_label}={arr2[i]:.3g}"))
        fig.update_layout(
            title=f"{x_label} response at fixed {y_label}",
            xaxis_title=x_label, yaxis_title=z_label, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with sc2:
        fig = go.Figure()
        for i in slices:
            fig.add_trace(go.Scatter(
                x=arr2, y=Z[:, i], mode="lines",
                name=f"{x_label}={arr1[i]:.3g}"))
        fig.update_layout(
            title=f"{y_label} response at fixed {x_label}",
            xaxis_title=y_label, yaxis_title=z_label, height=380)
        st.plotly_chart(fig, use_container_width=True)
