# src/views/synergy_view.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from logic.data_processing import calculate_high_throughput_synergy
from logic.plotting import calculate_synergy_surface_grid, plot_response_surface, predict_surface
from logic.null_surface import reference_surface, shared_baseline
from utils.ui_helpers import format_variable_options

# Diverging colorscale for difference-to-neutral: green = synergy (Δ < 0),
# near-white = additive (Δ ≈ 0), red = antagonism (Δ > 0).
_DELTA_COLORSCALE = [
    [0.0, "rgb(0,140,0)"],
    [0.5, "rgb(245,245,245)"],
    [1.0, "rgb(200,0,0)"],
]

def render():
    """Renders the Row-Based High-Throughput Synergy Analysis Tab."""
    st.subheader("🧪 High-Throughput Synergy Screening")
    
    if 'analysis_done' not in st.session_state or not st.session_state.analysis_done:
        st.info("Please load a project and run an analysis first (to enable AI-assisted reference predictions).")
        return

    st.markdown("""
    This tool analyzes drug combinations row-by-row. It supports **3+ drug combinations** and **sparse datasets**.
    If specific single-agent reference doses are missing, it uses your trained AI model to estimate them.
    """)

    with st.container(border=True):
        st.markdown("##### ⚙️ Screening Configuration")

        col1, col2 = st.columns(2)

        with col1:
            drug_options = st.session_state.independent_vars
            selected_drugs = st.multiselect(
                "Select All Drug Columns (Dose)",
                options=drug_options,
                default=drug_options[:2] if len(drug_options) >= 2 else drug_options
            )

        with col2:
            st.info("Each model will automatically be scored against its own response column.")

        st.info("Analysis will run for **all available models**. Each model scores synergy for its own response variable and provides AI imputation for any missing single-agent controls.")

    if st.button("🚀 Run Screening Analysis", type="primary"):
        if not selected_drugs:
            st.error("Please select at least two drug columns.")
            return

        wrapped_models = st.session_state.get('wrapped_models', {})
        if not wrapped_models:
            st.error("No models found. Please run an analysis first.")
            return

        all_synergy_results = {}
        skipped = []
        with st.spinner("Analyzing combinations for all models..."):
            try:
                for mname, mobj in wrapped_models.items():
                    if mname not in st.session_state.exp_df.columns:
                        skipped.append(mname)
                        continue
                    df_results, direction = calculate_high_throughput_synergy(
                        df=st.session_state.exp_df,
                        dose_cols=selected_drugs,
                        effect_col=mname,
                        model=mobj
                    )
                    all_synergy_results[mname] = (df_results, direction, mname)

                if not all_synergy_results:
                    st.error("No models matched a response column in the data.")
                    return

                st.session_state.synergy_results_all = all_synergy_results
                st.session_state.synergy_active_drugs = selected_drugs
                direction_sample = next(iter(all_synergy_results.values()))[1]
                msg = f"Analysis Complete! Auto-detected Data Type: **{direction_sample.title()}**"
                if skipped:
                    msg += f" _(Skipped: {', '.join(skipped)} — no matching data column)_"
                st.success(msg)

            except Exception as e:
                st.error(f"Analysis Failed: {e}")
                st.exception(e)

    # --- RESULTS DISPLAY ---
    if st.session_state.get('synergy_results_all'):
        all_results = st.session_state.synergy_results_all
        active_drugs = st.session_state.get('synergy_active_drugs', [])

        st.write("---")
        st.subheader("📊 Analysis Results")

        filter_col, _ = st.columns([1, 2])
        show_only_combos = filter_col.checkbox("Hide Single Agents / Controls", value=True)

        def color_synergy(val):
            if pd.isna(val): return ''
            if val > 0.1: return 'background-color: #d4edda; color: #155724'
            if val < -0.1: return 'background-color: #f8d7da; color: #721c24'
            return ''

        def color_zscore(val):
            if pd.isna(val): return ''
            if val > 2.0: return 'background-color: #d4edda; color: #155724'
            if val < -2.0: return 'background-color: #f8d7da; color: #721c24'
            return ''

        # Per-model tables
        for model_name, (results, direction, effect_col) in all_results.items():
            display_df = results.copy()
            if show_only_combos:
                display_df = display_df[display_df['Analysis_Note'] != 'Single Agent / Control']

            with st.expander(f"📋 {model_name}  ·  Response: {effect_col}  ·  {direction.title()}", expanded=True):
                combo_rows = display_df[display_df['Analysis_Note'] != 'Single Agent / Control']
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Combinations", len(combo_rows))
                m2.metric("Synergistic Hits (HSA > 0.1)", len(combo_rows[combo_rows['HSA_Synergy'] > 0.1]))
                m3.metric("Antagonistic Hits (HSA < -0.1)", len(combo_rows[combo_rows['HSA_Synergy'] < -0.1]))

                raw_cols = [c for c in ['HSA_Synergy', 'Bliss_Synergy', 'ZIP_Synergy'] if c in display_df.columns]
                z_cols   = [c for c in ['HSA_Synergy_Z', 'Bliss_Synergy_Z', 'ZIP_Synergy_Z'] if c in display_df.columns]
                fmt = {c: '{:.4f}' for c in raw_cols + z_cols + ['std_effect'] if c in display_df.columns}
                st.dataframe(
                    display_df.style
                        .map(color_synergy, subset=raw_cols)
                        .map(color_zscore,  subset=z_cols)
                        .format(fmt),
                    width="stretch",
                    height=400
                )

                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Results CSV", csv,
                    f"synergy_{model_name}.csv", "text/csv",
                    key=f"dl_synergy_{model_name.replace(' ', '_')}"
                )

        # --- Visualization (shared, with model picker) ---
        st.write("---")
        st.subheader("📈 Visualization")

        model_names = list(all_results.keys())
        viz_model_name = st.selectbox("Select model to visualize", model_names, key="synergy_viz_model")
        viz_results, _, viz_effect_col = all_results[viz_model_name]
        viz_display = viz_results.copy()
        if show_only_combos:
            viz_display = viz_display[viz_display['Analysis_Note'] != 'Single Agent / Control']

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Distribution", "Scatter Plot", "🧊 3D Synergy Surface", "🔀 Cross-Model Comparison", "🧬 Null Surface (Δ)", "🔬 Mechanistic"])

        with tab1:
            fig_hist = px.histogram(
                viz_display,
                x="HSA_Synergy",
                nbins=30,
                title=f"Distribution of Synergy Scores (HSA) — {viz_model_name}",
                color_discrete_sequence=['#636EFA']
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig_hist, width="stretch")

        with tab2:
            viz_display['Total_Dose'] = viz_display[active_drugs].sum(axis=1)
            fig_scat = px.scatter(
                viz_display,
                x="Total_Dose",
                y="HSA_Synergy",
                color="Analysis_Note",
                hover_data=active_drugs,
                title=f"Total Dose vs. Synergy Score — {viz_model_name}"
            )
            fig_scat.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig_scat, width="stretch")

        with tab3:
            st.markdown("**Visualize Interaction Surface:** Shape = Effect, Color = Synergy")

            if len(active_drugs) < 2:
                st.warning("Need at least 2 drugs for a surface plot.")
            else:
                c1, c2, c3 = st.columns(3)
                d1 = c1.selectbox("X-Axis Drug", active_drugs, index=0, key="surf_d1")
                d2 = c2.selectbox("Y-Axis Drug", active_drugs, index=1 if len(active_drugs) > 1 else 0, key="surf_d2")
                method = c3.radio("Synergy Method", ["HSA", "Bliss"], index=0, key="surf_method")

                model_obj = st.session_state.wrapped_models.get(viz_model_name)
                if not model_obj:
                    st.error("Model not found. Please re-run analysis.")
                else:
                    if st.button("Draw Surface"):
                        max_d1 = st.session_state.exp_df[d1].max()
                        max_d2 = st.session_state.exp_df[d2].max()
                        fixed_vars = {d: 0.0 for d in st.session_state.independent_vars if d not in [d1, d2]}

                        with st.spinner("Calculating synergy surface..."):
                            x_g, y_g, z_g, syn_g = calculate_synergy_surface_grid(
                                model=model_obj,
                                drug1=d1, drug2=d2,
                                fixed_vars=fixed_vars,
                                range1=(0, max_d1),
                                range2=(0, max_d2),
                                method=method.lower()
                            )
                            d1_desc = st.session_state.variable_descriptions.get(d1, d1)
                            d2_desc = st.session_state.variable_descriptions.get(d2, d2)
                            fig_surf = plot_response_surface(
                                dataframe=st.session_state.exp_df,
                                OLS_model_1=model_obj,
                                all_alphabet_vars=st.session_state.independent_vars,
                                x_var=d1, y_var=d2,
                                z_var_1=viz_effect_col,
                                fixed_vars_dict_1=fixed_vars,
                                variable_descriptions=st.session_state.variable_descriptions,
                                show_actual_data=False,
                                x_grid_override=x_g, y_grid_override=y_g, z_grid_override=z_g,
                                surfacecolor=syn_g,
                                surfacecolor_label=f"Synergy ({method.upper()})",
                                main_title=f"3D Synergy Surface: {d1_desc} vs {d2_desc}",
                                z_title="Inhibition (Predicted)",
                                x_title=d1_desc, y_title=d2_desc,
                            )
                            st.plotly_chart(fig_surf, width="stretch")

        with tab4:
            st.markdown("#### Cross-Model Synergy Comparison (Z-Score Heatmap)")
            st.markdown("""
            Z-scores normalise each model's synergy distribution to **mean = 0, std = 1**,
            putting all response variables on the same scale.
            A combination scoring **Z > +2** across multiple models is robustly synergistic
            regardless of which biological endpoint you look at.
            - 🟢 **Z > +2** — significantly synergistic
            - 🔴 **Z < −2** — significantly antagonistic
            """)

            if len(all_results) < 2:
                st.info("Run analysis with at least 2 models to enable cross-model comparison.")
            else:
                z_metric = st.selectbox(
                    "Synergy metric for comparison",
                    ["HSA_Synergy_Z", "Bliss_Synergy_Z", "ZIP_Synergy_Z"],
                    key="compare_metric"
                )

                # Build combo label from drug doses
                exp_df = st.session_state.exp_df
                combo_labels = {}
                for idx, row in exp_df.iterrows():
                    active = [d for d in active_drugs if row[d] > 1e-6]
                    if len(active) >= 2:
                        combo_labels[idx] = " + ".join(
                            [f"{st.session_state.variable_descriptions.get(d, d)}({row[d]:.2g})" for d in active]
                        )

                compare_data = {}
                for mname, (res, _, _eff) in all_results.items():
                    combo_rows = res[res['Analysis_Note'] != 'Single Agent / Control']
                    if z_metric in combo_rows.columns:
                        compare_data[mname] = combo_rows[z_metric]

                if not compare_data:
                    st.warning("Z-score columns not found — re-run the screening analysis.")
                else:
                    compare_df = pd.DataFrame(compare_data)
                    compare_df.index = [combo_labels.get(i, str(i)) for i in compare_df.index]
                    compare_df = compare_df.dropna(how='all')

                    if compare_df.empty:
                        st.warning("No combination rows with valid Z-scores found.")
                    else:
                        fig_heat = px.imshow(
                            compare_df,
                            color_continuous_scale='RdBu_r',
                            color_continuous_midpoint=0,
                            zmin=-3, zmax=3,
                            title=f"Cross-Model Z-Score Heatmap  ({z_metric.replace('_Z', '').replace('_', ' ')})",
                            labels={'color': 'Z-Score', 'x': 'Model / Response Variable', 'y': 'Drug Combination'},
                            aspect='auto'
                        )
                        fig_heat.update_layout(height=max(400, len(compare_df) * 30 + 100))
                        st.plotly_chart(fig_heat, width="stretch")

                        # Summary: how many combos are consistently synergistic
                        threshold = st.slider("Consistency threshold (|Z| ≥)", 1.0, 3.0, 2.0, 0.5, key="z_thresh")
                        n_models = compare_df.shape[1]
                        consistent_syn = compare_df[(compare_df >= threshold).sum(axis=1) == n_models]
                        consistent_ant = compare_df[(compare_df <= -threshold).sum(axis=1) == n_models]

                        c1, c2 = st.columns(2)
                        c1.metric(f"Consistently Synergistic (Z ≥ {threshold} in ALL models)", len(consistent_syn))
                        c2.metric(f"Consistently Antagonistic (Z ≤ −{threshold} in ALL models)", len(consistent_ant))

                        if not consistent_syn.empty:
                            st.markdown("**Consistently Synergistic Combinations:**")
                            st.dataframe(consistent_syn.style.format("{:.2f}"), width="stretch")
                        if not consistent_ant.empty:
                            st.markdown("**Consistently Antagonistic Combinations:**")
                            st.dataframe(consistent_ant.style.format("{:.2f}"), width="stretch")

        with tab5:
            st.markdown(
                "**Difference-to-Neutral:** the model surface vs the non-interaction *null* "
                "(Bliss / HSA / Loewe) built from monotherapy Hill fits. "
                "Color = Δ (model − null); for viability, **Δ < 0 = synergy** "
                "(combination kills more than the null predicts)."
            )

            hill_all = st.session_state.get('hill_fits') or {}
            hf = hill_all.get(viz_effect_col)
            usable = {d: f for d, f in hf.items() if d in active_drugs} if hf else {}

            if len(usable) < 2:
                st.info(
                    "Needs fitted monotherapy Hill curves for ≥2 drugs in this response. "
                    "Add single-drug rows (one drug varied, the others at 0) so the marginals "
                    "can be fit — none were detected for this model."
                )
            else:
                drugs_with_fit = list(usable.keys())
                c1, c2, c3 = st.columns(3)
                nd1 = c1.selectbox("X-Axis Drug", drugs_with_fit, index=0, key="null_d1")
                nd2 = c2.selectbox("Y-Axis Drug", drugs_with_fit,
                                   index=1 if len(drugs_with_fit) > 1 else 0, key="null_d2")
                nmethod = c3.radio("Null model", ["Bliss", "HSA", "Loewe"], index=0, key="null_method")

                model_obj = st.session_state.wrapped_models.get(viz_model_name)
                if nd1 == nd2:
                    st.warning("Pick two different drugs.")
                elif not model_obj:
                    st.error("Model not found. Please re-run analysis.")
                elif st.button("Draw Δ Surface", key="null_draw"):
                    max_d1 = st.session_state.exp_df[nd1].max()
                    max_d2 = st.session_state.exp_df[nd2].max()
                    fixed_vars = {d: 0.0 for d in st.session_state.independent_vars if d not in [nd1, nd2]}
                    with st.spinner("Computing null surface and Δ..."):
                        # Raw viability model surface — same units as the null surface.
                        # (NOT calculate_synergy_surface_grid, which clips to 0-1 inhibition.)
                        x_g, y_g, z_g = predict_surface(
                            model_obj, st.session_state.independent_vars,
                            nd1, nd2, fixed_vars, (0, max_d1), (0, max_d2))
                        rows = [{**fixed_vars, nd1: float(x_g[i, j]), nd2: float(y_g[i, j])}
                                for i in range(x_g.shape[0]) for j in range(x_g.shape[1])]
                        ref = reference_surface(nmethod.lower(), rows, usable).reshape(x_g.shape)

                        # Clamp the model surface to the physical response range
                        # [Einf, E0] (from the Hill baselines). A polynomial model can
                        # extrapolate to impossible values (e.g. viability < 0), which
                        # would falsely inflate synergy — clamp for both Δ and display.
                        e0_bar, einf_bar = shared_baseline(usable)
                        lo, hi = (einf_bar, e0_bar) if einf_bar <= e0_bar else (e0_bar, einf_bar)
                        n_out = int(np.sum((z_g < lo) | (z_g > hi)))
                        z_show = np.clip(z_g, lo, hi)
                        delta = z_show - ref
                        cmax = float(np.nanmax(np.abs(delta))) or 1.0

                        if n_out:
                            st.caption(
                                f"⚠️ The model surface extrapolated outside the physical range "
                                f"[{lo:.0f}, {hi:.0f}] in {n_out / z_g.size * 100:.0f}% of the grid "
                                f"— clamped for Δ and display. A bounded / mechanistic model "
                                f"(Phase 2) avoids this."
                            )

                        d1_desc = st.session_state.variable_descriptions.get(nd1, nd1)
                        d2_desc = st.session_state.variable_descriptions.get(nd2, nd2)
                        fig = plot_response_surface(
                            dataframe=st.session_state.exp_df, OLS_model_1=model_obj,
                            all_alphabet_vars=st.session_state.independent_vars,
                            x_var=nd1, y_var=nd2, z_var_1=viz_effect_col,
                            fixed_vars_dict_1=fixed_vars,
                            variable_descriptions=st.session_state.variable_descriptions,
                            show_actual_data=False,
                            x_grid_override=x_g, y_grid_override=y_g, z_grid_override=z_show,
                            surfacecolor=delta, surfacecolor_colorscale=_DELTA_COLORSCALE,
                            surfacecolor_cmin=-cmax, surfacecolor_cmax=cmax,
                            surfacecolor_label=f"Δ vs {nmethod} (green = synergy)",
                            main_title=f"Model surface — Δ vs {nmethod}",
                            z_title=viz_effect_col, x_title=d1_desc, y_title=d2_desc,
                        )
                        fig_null = plot_response_surface(
                            dataframe=st.session_state.exp_df, OLS_model_1=model_obj,
                            all_alphabet_vars=st.session_state.independent_vars,
                            x_var=nd1, y_var=nd2, z_var_1=viz_effect_col,
                            fixed_vars_dict_1=fixed_vars,
                            variable_descriptions=st.session_state.variable_descriptions,
                            show_actual_data=False,
                            x_grid_override=x_g, y_grid_override=y_g, z_grid_override=ref,
                            main_title=f"Null surface ({nmethod})",
                            z_title=viz_effect_col, x_title=d1_desc, y_title=d2_desc,
                        )
                        cM, cN = st.columns(2)
                        cM.plotly_chart(fig, width="stretch")
                        cN.plotly_chart(fig_null, width="stretch")

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Mean Δ", f"{np.nanmean(delta):.2f}")
                        m2.metric("Min Δ (peak synergy)", f"{np.nanmin(delta):.2f}")
                        m3.metric("Synergistic area (Δ < 0)", f"{np.mean(delta < 0) * 100:.0f}%")

        with tab6:
            mech = {n: m for n, m in st.session_state.get('wrapped_models', {}).items()
                    if getattr(m, 'IS_MECHANISTIC', False)}
            if not mech:
                st.info(
                    "Parameter-based synergy appears here when you fit a **MuSyC** or "
                    "**BRAID** model (Library → *Run a New Analysis* → choose MuSyC / BRAID). "
                    "These read synergy **straight from the fitted surface** — α/β/γ (MuSyC) "
                    "or κ (BRAID) — instead of subtracting a null."
                )
            else:
                st.markdown(
                    "Synergy read **directly from the fitted mechanistic surface** "
                    "(no null subtraction). Conventions follow the `synergy` package: "
                    "α / γ compare to 1, β / κ compare to 0."
                )

                def _color_verdict(v):
                    return ('color: green; font-weight: bold' if v == 'Synergistic'
                            else 'color: #c00; font-weight: bold' if v == 'Antagonistic'
                            else 'color: gray')

                for name, model in mech.items():
                    r2 = model.r2_score if model.r2_score is not None else float('nan')
                    with st.expander(f"🔬 {name} · {model.kind.upper()}  (R² = {r2:.3f})", expanded=True):
                        metrics = model.synergy_metrics()
                        mdf = pd.DataFrame(metrics)
                        st.dataframe(
                            mdf.style.map(_color_verdict, subset=['Verdict']).format(
                                {'Value': '{:.3f}', 'Baseline': '{:.0f}'}),
                            width="stretch", hide_index=True,
                        )
                        syn = [r['Parameter'].split(' ')[0] for r in metrics if r['Verdict'] == 'Synergistic']
                        if syn:
                            st.success(f"Synergistic on: **{', '.join(syn)}**")
                        else:
                            st.info("No clear synergy axis — mostly additive / antagonistic.")
