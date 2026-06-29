# src/views/optimizer_view.py
import streamlit as st
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime
import json
from logic.optimization import (run_multi_objective_penalty_optimization,
                                  run_weighted_tradeoff_optimization,
                                  difference_objective_function, run_optimization,
                                  run_grid_search_optimization, run_classic_multi_objective_optimization,
                                  objective_function)
from logic.data_processing import generate_optimization_report
from logic.hill_fit import compute_dri
from logic.plotting import plot_response_curve
from utils.ui_helpers import (get_optimization_bounds_and_algo,
                              format_variable_options, display_surface_plot)
from utils.state_management import clear_optimizer_results

def load_config():
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return a default configuration if the file doesn't exist or is empty/corrupt
        return {
            "multi_opt_tolerance": 0.1 
        }

def render():
    """Renders all UI components and logic for the Optimizer tab."""
    st.subheader("🎯 Optimizer")

    optimizer_types = ["Single-Objective", "Multi-Objective"]

    opt_type = st.radio(
        "Select Optimization Type",
        optimizer_types,
        horizontal=True,
        key="optimizer_type",
        on_change=clear_optimizer_results
    )

    if not st.session_state.get('wrapped_models'):
        st.info("Please analyze a project in the 'Project Library' before using the optimizer.")
        return

    formatted_models = format_variable_options(st.session_state.wrapped_models.keys())

    if opt_type == "Single-Objective":
        render_single_objective_optimizer(formatted_models)
    elif opt_type == "Multi-Objective":
        render_multi_objective_optimizer(formatted_models)

    render_optimizer_results()


def render_single_objective_optimizer(formatted_models):
    """Renders the UI and runs the logic for single-objective optimization."""
    selected_model_formatted = st.selectbox("Select model to optimize", options=formatted_models, key="single_opt_model", on_change=clear_optimizer_results)
    model_to_optimize = selected_model_formatted.split(":")[0]
    target_value = st.number_input("Target Value", value=0.0, format="%.4f", help="The optimizer will first try to find a solution that results in this exact value.", key="single_opt_target", on_change=clear_optimizer_results)
    fallback_goal = st.radio("If an exact match isn't possible, find the closest:", ("Value above target", "Value below target"), horizontal=True, key="single_opt_fallback", on_change=clear_optimizer_results)

    discrete_vars_single = st.multiselect(
        "Select any variables that are discrete (0/1 only)",
        options=st.session_state.detected_binary_vars,
        default=st.session_state.detected_binary_vars,
        key="single_opt_discrete_vars",
        help="An equality constraint (x * (1-x) = 0) will be added for these variables. This method is only suitable for binary (0/1) variables."
    )

    bounds, start_points, algorithm, algo_params = get_optimization_bounds_and_algo("single_obj")

    if st.button("Run Single-Objective Optimization", type="primary"):
        clear_optimizer_results()
        selected_model = st.session_state.wrapped_models[model_to_optimize]

        binary_constraints = []
        for i, var_name in enumerate(st.session_state.independent_vars):
            if var_name in discrete_vars_single:
                constraint_func = lambda x, index=i: x[index] * (1 - x[index])
                binary_constraints.append({'type': 'eq', 'fun': constraint_func})

        result = None
        with st.spinner("Attempting to find an exact match for the target value..."):
            obj_fun_exact = lambda x: abs(objective_function(x, selected_model, st.session_state.independent_vars) - target_value)
            result = run_optimization(obj_fun_exact, bounds, start_points, binary_constraints, algorithm, algo_params)

        if not result.success or result.fun > 1e-6:
            st.warning(f"Could not find an exact match for {target_value}. Now searching for the best alternative...")
            with st.spinner(f"Searching for the best value {fallback_goal.lower()}..."):
                fallback_constraints = list(binary_constraints)
                if fallback_goal == "Value above target":
                    obj_fun_fallback = lambda x: objective_function(x, selected_model, st.session_state.independent_vars)
                    cons_fallback = {'type': 'ineq', 'fun': lambda x: objective_function(x, selected_model, st.session_state.independent_vars) - target_value}
                    fallback_constraints.append(cons_fallback)
                else: # Value below target
                    obj_fun_fallback = lambda x: -objective_function(x, selected_model, st.session_state.independent_vars)
                    cons_fallback = {'type': 'ineq', 'fun': lambda x: target_value - objective_function(x, selected_model, st.session_state.independent_vars)}
                    fallback_constraints.append(cons_fallback)
                result = run_optimization(obj_fun_fallback, bounds, start_points, fallback_constraints, algorithm, algo_params)

        if result.success:
            st.success("Optimization Successful!")
            final_outcome = objective_function(result.x, selected_model, st.session_state.independent_vars)
            st.session_state.single_opt_results = {
                "dosages": result.x,
                "outcome": final_outcome,
                "model_name": model_to_optimize
            }
            report_data = {
                "optimization_type": "Single-Objective", "model_name": model_to_optimize,
                "summary_text": selected_model.get_summary(),
                "settings": {
                    "Algorithm": algorithm, "Target Value": target_value, "Fallback Goal": fallback_goal,
                    "Bounds": dict(zip(st.session_state.independent_vars, bounds)),
                },
                "results": {
                    "Status": "Success", "Final Outcome": final_outcome,
                    "Optimal Dosages": dict(zip(st.session_state.independent_vars, result.x))
                }
            }
            st.session_state.single_opt_report_data = report_data
            st.rerun()
        else:
            st.error(f"Optimization failed: {result.message}.")
            st.session_state.single_opt_results = None


def render_multi_objective_optimizer(formatted_models):
    """Renders the UI for multi-objective optimization, showing options based on unlock status."""
    st.subheader("Multi-Objective Optimization")

    if st.session_state.get('experimental_unlocked', False):
        multi_obj_method = st.radio(
            "Select Multi-Objective Method",
            ("Classic Two-Stage", "Weighted Score (Experimental)"),
            horizontal=True,
            key="multi_obj_method",
            help="**Classic Two-Stage**: The original auto-relaxing, dose-efficiency method. **Weighted Score**: A modern approach that balances multiple goals."
        )
    else:
        multi_obj_method = "Classic Two-Stage"

    if multi_obj_method.startswith("Weighted Score"):
        render_weighted_score_ui(formatted_models)
    else:
        render_classic_two_stage_ui(formatted_models)


def render_weighted_score_ui(formatted_models):
    """Renders the UI for the modern weighted score method with dual-range constraints."""
    st.info(
        """
        This tool finds an optimal solution by constraining **BOTH** models to specific ranges 
        and then finding the best trade-off (Weighted Score) within those limits.
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Goal 1 (Model A)**")
        model_1_formatted = st.selectbox("Select Model A", options=formatted_models, key="adv_model_1", on_change=clear_optimizer_results)
        model_1_name = model_1_formatted.split(":")[0]

        r_min_1 = st.number_input(f"Min value for {model_1_name}", value=0.0, format="%.4f", key="rmin1")
        r_max_1 = st.number_input(f"Max value for {model_1_name}", value=1.0, format="%.4f", key="rmax1")

    with c2:
        st.write("**Goal 2 (Model B)**")
        model_2_options = [m for m in formatted_models if not m.startswith(model_1_name)]
        if not model_2_options:
            st.warning("You need at least two models to perform this analysis.")
            return

        model_2_formatted = st.selectbox("Select Model B", options=model_2_options, key="adv_model_2", on_change=clear_optimizer_results)
        model_2_name = model_2_formatted.split(":")[0]
        
        # New: Range inputs for Model 2
        r_min_2 = st.number_input(f"Min value for {model_2_name}", value=0.0, format="%.4f", key="rmin2")
        r_max_2 = st.number_input(f"Max value for {model_2_name}", value=1.0, format="%.4f", key="rmax2")

    with st.expander("Weighted Score Configuration"):
        st.write("Configure the weights to define the 'Best Trade-off'. Higher weight = more priority.")
        w1 = st.slider(f"Weight for {model_1_name}", 0.0, 1.0, 0.5, 0.05)
        w2 = st.slider(f"Weight for {model_2_name}", 0.0, 1.0, 0.5, 0.05)
        w_dosage = st.slider("Weight for Total Dosage (Penalty)", 0.0, 1.0, 0.1, 0.05)

    st.write("---")
    st.write("**Define Search Space and Algorithm**")
    bounds, start_points, algorithm, algo_params = get_optimization_bounds_and_algo("advanced_tradeoff")

    if st.button("Run Multi-Objective Optimization", type="primary"):
        clear_optimizer_results()
        model_1 = st.session_state.wrapped_models[model_1_name]
        model_2 = st.session_state.wrapped_models[model_2_name]
        independent_vars = st.session_state.independent_vars

        # Stage 1: Find Feasible Region
        with st.spinner("Stage 1/3: Finding feasible solution within ranges..."):
            constrained_result = run_multi_objective_penalty_optimization(
                model_1, model_2, independent_vars, bounds, start_points, 
                r_min_1, r_max_1, r_min_2, r_max_2,
                algorithm, algo_params
            )
            if not constrained_result.success or constrained_result.fun > 0.1: 
                # If penalty > 0.1, it implies we couldn't satisfy the ranges
                st.warning("Stage 1 Warning: Could not find a solution strictly inside both ranges. Proceeding with best attempt...")

        # Stage 2: Grid Search (Exploration)
        with st.spinner("Stage 2/3: Exploring weighted solutions..."):
            config = load_config()
            tolerance = config.get("multi_opt_tolerance", 0.1)
            weights = {"model_1": w1, "model_2": w2, "total_dosage": w_dosage}
            
            # Note: Grid search primarily slides along Model 1. 
            # It may produce points outside Model 2's range, but gives good context.
            top_5_weighted = run_grid_search_optimization(
                model_1, model_2, independent_vars, bounds, start_points, 
                algorithm, algo_params, constrained_result, weights, tolerance
            )

        # Stage 3: Strict Trade-off Optimization
        with st.spinner("Stage 3/3: Finding best strict trade-off..."):
            tradeoff_result = run_weighted_tradeoff_optimization(
                model_1, model_2, independent_vars, bounds, start_points,
                r_min_1, r_max_1, r_min_2, r_max_2,
                weights, algorithm, algo_params
            )
            
            candidate_B = None
            if tradeoff_result.success:
                candidate_B = {
                    "dosages": tradeoff_result.x,
                    "outcome_1": objective_function(tradeoff_result.x, model_1, independent_vars),
                    "outcome_2": objective_function(tradeoff_result.x, model_2, independent_vars)
                }
            else:
                st.warning("Could not find a valid trade-off solution within the strict constraints.")

        st.success("Multi-Objective Analysis Complete!")
        st.session_state.advanced_tradeoff_results = {
            "top_5_weighted": top_5_weighted,
            "candidate_B": candidate_B,
            "model_1_name": model_1_name,
            "model_2_name": model_2_name,
        }
        st.rerun()

def render_classic_two_stage_ui(formatted_models):
    """Renders the UI for the restored classic two-stage optimization."""
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Goal 1 (Constraining Model)**")
        model_1_formatted = st.selectbox("Select Model to Constrain", options=formatted_models, key="classic_model_1", on_change=clear_optimizer_results)
        model_1_name = model_1_formatted.split(":")[0]

        r_min = st.number_input(f"Minimum acceptable value for {model_1_name}", value=0.0, format="%.4f")
        r_max = st.number_input(f"Maximum acceptable value for {model_1_name}", value=0.0, format="%.4f")

    with c2:
        st.write("**Goal 2 (Optimizing Model)**")
        model_2_options = [m for m in formatted_models if not m.startswith(model_1_name)]
        if not model_2_options:
            st.warning("You need at least two models to perform this analysis.")
            return

        model_2_formatted = st.selectbox("Select Model to Optimize", options=model_2_options, key="classic_model_2", on_change=clear_optimizer_results)
        model_2_name = model_2_formatted.split(":")[0]

        target_model_goal = st.radio("Optimization Goal", ("Maximize", "Minimize"), key="classic_multi_opt_goal", horizontal=True)

    st.write("---")
    st.write("**Define Search Space and Algorithm**")
    bounds, start_points, algorithm, algo_params = get_optimization_bounds_and_algo("classic_multi", default_algo='SLSQP (Local)')

    if st.button("Run optimization", type="primary"):
        clear_optimizer_results()
        model_1 = st.session_state.wrapped_models[model_1_name]
        model_2 = st.session_state.wrapped_models[model_2_name]

        with st.spinner("Running Classic Two-Stage Optimization..."):
            final_dosages, status_message = run_classic_multi_objective_optimization(
                model_1, model_2, st.session_state.independent_vars, bounds, start_points,
                r_min, r_max, target_model_goal, algorithm, algo_params
            )

        if final_dosages is not None:
            st.success("Classic Two-Stage Optimization Successful!")
            outcome_1 = objective_function(final_dosages, model_1, st.session_state.independent_vars)
            outcome_2 = objective_function(final_dosages, model_2, st.session_state.independent_vars)
            
            st.session_state.classic_multi_opt_results = {
                "dosages": final_dosages,
                "outcome_1": outcome_1,
                "outcome_2": outcome_2,
                "model_1_name": model_1_name,
                "model_2_name": model_2_name
            }
            st.rerun()
        else:
            st.error(f"Optimization failed: {status_message}")


def render_optimizer_results():
    """Renders the results section, which displays output from any of the optimizers."""
    active_results = None
    res = None
    report_data_to_use = None

    if st.session_state.get("single_opt_results"):
        active_results = "single"
        res = st.session_state.single_opt_results
        report_data_to_use = st.session_state.get("single_opt_report_data")
    elif st.session_state.get("advanced_tradeoff_results"):
        active_results = "advanced_tradeoff"
        res = st.session_state.advanced_tradeoff_results
    elif st.session_state.get("classic_multi_opt_results"):
        active_results = "classic_multi"
        res = st.session_state.classic_multi_opt_results

    if not active_results:
        return

    st.header("Optimization Result")

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Clear Results"):
            clear_optimizer_results()
            st.rerun()

    descriptions = st.session_state.variable_descriptions
    descriptive_vars = [descriptions.get(var, var) for var in st.session_state.independent_vars]

    if active_results == "single":
        dosages_df = pd.DataFrame({'Variable': descriptive_vars, 'Optimal Value': res['dosages']})
        model_desc = descriptions.get(res['model_name'], res['model_name'])
        st.metric(label=f"Optimal {model_desc} Value", value=f"{res['outcome']:.4f}")
        st.write("Optimal Dosages:")
        st.dataframe(dosages_df, width="stretch")
        render_dri_panel(
            model_name=res['model_name'],
            dosages_dict=dict(zip(st.session_state.independent_vars, res['dosages'])),
            outcome=res['outcome'],
        )
        render_result_visualization(active_results, res)

    elif active_results == "advanced_tradeoff":
        st.subheader("Analysis Results")
        model_1_name = res['model_1_name']
        model_2_name = res['model_2_name']
        model_1_desc = descriptions.get(model_1_name, model_1_name)
        model_2_desc = descriptions.get(model_2_name, model_2_name)
        top_5_weighted = res.get('top_5_weighted')

        with st.container(border=True):
            st.markdown("#### Optimal Solution (Weighted Score)")
            st.info(f"This solution finds the best outcome for **{model_2_desc}** while keeping **{model_1_desc}** within its constraints, ranked by a weighted score.")
            if top_5_weighted:
                top_solution = top_5_weighted[0]
                outcome_1 = objective_function(top_solution.x, st.session_state.wrapped_models[model_1_name], st.session_state.independent_vars)
                outcome_2 = objective_function(top_solution.x, st.session_state.wrapped_models[model_2_name], st.session_state.independent_vars)
                total_dosage = sum(top_solution.x)
                difference = outcome_1 - outcome_2

                c1, c2, c3, c4 = st.columns(4)
                c1.metric(label=f"Outcome: {model_1_desc}", value=f"{outcome_1:.4f}")
                c2.metric(label=f"Outcome: {model_2_desc}", value=f"{outcome_2:.4f}")
                c3.metric(label="Total Dosage", value=f"{total_dosage:.4f}")
                c4.metric(label="Difference", value=f"{difference:.4f}")

                st.write("**Optimal Dosages for this Solution:**")
                dosages_df = pd.DataFrame({'Variable': descriptive_vars, 'Optimal Value': top_solution.x})
                st.dataframe(dosages_df, width="stretch")
            else:
                st.warning("No weighted solutions were found within the specified tolerance.")

        render_result_visualization(active_results, res)

        with st.expander("Supporting Information"):
            if top_5_weighted:
                st.write("#### Top 5 Weighted Solutions")
                for i, result in enumerate(top_5_weighted):
                    st.write(f"---")
                    st.write(f"#### Rank {i+1} Solution")
                    outcome_1 = objective_function(result.x, st.session_state.wrapped_models[model_1_name], st.session_state.independent_vars)
                    outcome_2 = objective_function(result.x, st.session_state.wrapped_models[model_2_name], st.session_state.independent_vars)
                    total_dosage = sum(result.x)
                    difference = outcome_1 - outcome_2

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(label=f"Outcome: {model_1_desc}", value=f"{outcome_1:.4f}")
                    c2.metric(label=f"Outcome: {model_2_desc}", value=f"{outcome_2:.4f}")
                    c3.metric(label="Total Dosage", value=f"{total_dosage:.4f}")
                    c4.metric(label="Difference", value=f"{difference:.4f}")

                    st.write("**Optimal Dosages for this Solution:**")
                    dosages_df = pd.DataFrame({'Variable': descriptive_vars, 'Optimal Value': result.x})
                    st.dataframe(dosages_df, width="stretch")

            if res.get("candidate_B"):
                st.write("---")
                st.write("#### Best Trade-off Solution")
                cand_B = res['candidate_B']
                st.info(f"This solution finds the greatest positive difference between the two models, based on your Goal 2 selection.")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(label=f"Outcome: {model_1_desc}", value=f"{cand_B['outcome_1']:.4f}")
                c2.metric(label=f"Outcome: {model_2_desc}", value=f"{cand_B['outcome_2']:.4f}")
                c3.metric(label="Total Dosage", value=f"{sum(cand_B['dosages']):.4f}")
                c4.metric(label="Difference", value=f"{abs(cand_B['outcome_1'] - cand_B['outcome_2']):.4f}")
                st.write("**Optimal Dosages for this Solution:**")
                dosages_df_B = pd.DataFrame({'Variable': descriptive_vars, 'Optimal Value': cand_B['dosages']})
                st.dataframe(dosages_df_B, width="stretch")

    elif active_results == "classic_multi":
        model_1_name = res['model_1_name']
        model_2_name = res['model_2_name']
        model_1_desc = descriptions.get(model_1_name, model_1_name)
        model_2_desc = descriptions.get(model_2_name, model_2_name)

        c1, c2, c3 = st.columns(3)
        c1.metric(label=f"Outcome: {model_1_desc}", value=f"{res['outcome_1']:.4f}")
        c2.metric(label=f"Outcome: {model_2_desc}", value=f"{res['outcome_2']:.4f}")
        c3.metric(label="Total Dosage", value=f"{sum(res['dosages']):.4f}")

        st.write("**Optimal Dosages:**")
        dosages_df = pd.DataFrame({'Variable': descriptive_vars, 'Optimal Value': res['dosages']})
        st.dataframe(dosages_df, width="stretch")
        render_result_visualization(active_results, res)


def render_dri_panel(model_name: str, dosages_dict: dict, outcome: float):
    """
    Renders the Dose Reduction Index panel for a single-objective optimizer result.
    Only shown when monotherapy edge data was detected at analysis time.
    Wrapped in a collapsed expander — safe, never breaks the results above.
    """
    import plotly.graph_objects as go

    hill_fits = st.session_state.get("hill_fits") or {}
    drug_fits = hill_fits.get(model_name)

    if not drug_fits:
        return  # No monotherapy data — panel stays hidden

    independent_vars = st.session_state.independent_vars
    descriptions     = st.session_state.variable_descriptions

    # Only show panel for drugs that have both a Hill fit AND a non-zero combo dose
    eligible = {
        drug: hf for drug, hf in drug_fits.items()
        if drug in dosages_dict and dosages_dict[drug] > 1e-9
    }
    if not eligible:
        return

    with st.expander("📉 Dose Reduction Index (DRI)", expanded=True):
        st.caption(
            "Compares the optimal combination dose to the monotherapy dose required "
            "for the same predicted effect. Fitted from single-agent rows in your dataset."
        )

        # Warn for any drug with very few monotherapy points
        sparse_drugs = [
            descriptions.get(drug, drug)
            for drug, hf in eligible.items()
            if hf.n_points <= 3
        ]
        if sparse_drugs:
            st.warning(
                f"**Low data warning — {', '.join(sparse_drugs)}**: "
                "Hill curve fitted from only 3 single-agent points. "
                "The fit passes through all points exactly (R²=1.0 is not meaningful here). "
                "DRI values are directionally informative but imprecise — "
                "add more monotherapy doses (5+ recommended) for reliable estimates.",
                icon="⚠️",
            )

        dri_rows = []
        bar_names, bar_vals, bar_colors = [], [], []

        for drug, hf in eligible.items():
            d_combo  = dosages_dict[drug]
            drug_label = descriptions.get(drug, drug)
            dri_info = compute_dri(hf, d_combo, outcome)

            row = {
                "Drug":          drug_label,
                "Combo dose":    f"{d_combo:.4g}",
                "EC50 (mono)":   f"{hf.EC50:.4g}",
                "Hill slope h":  f"{hf.h:.2f}",
                "R² (Hill fit)": f"{hf.r2:.3f}" if not np.isnan(hf.r2) else "—",
            }

            if not dri_info["achievable"]:
                row["Mono equiv. dose"] = "Not achievable alone"
                row["DRI"] = "∞  (floor effect)"
                row["Interpretation"] = "Drug alone cannot reach this viability"
            elif dri_info["dri"] is None:
                row["Mono equiv. dose"] = f"{dri_info['mono_dose']:.4g}"
                row["DRI"] = "—"
                row["Interpretation"] = "Combo dose is zero"
            else:
                dri = dri_info["dri"]
                mono = dri_info["mono_dose"]
                row["Mono equiv. dose"] = f"{mono:.4g}"
                row["DRI"] = f"{dri:.2f}×"
                if dri > 1:
                    row["Interpretation"] = f"Combo uses {dri:.1f}× less {drug_label}"
                elif dri < 1:
                    row["Interpretation"] = f"Combo uses {1/dri:.1f}× MORE {drug_label}"
                else:
                    row["Interpretation"] = "No dose advantage"

                bar_names += [f"{drug_label}\n(alone)", f"{drug_label}\n(combo)"]
                bar_vals  += [mono, d_combo]
                bar_colors += ["#e74c3c", "#27ae60"]

            dri_rows.append(row)

        st.table(pd.DataFrame(dri_rows))

        if bar_names:
            fig = go.Figure(go.Bar(
                x=bar_names, y=bar_vals,
                marker_color=bar_colors,
                text=[f"{v:.3g}" for v in bar_vals],
                textposition="outside",
            ))
            fig.update_layout(
                title=f"Monotherapy vs Combination dose — same predicted effect ({outcome:.3g})",
                yaxis_title="Dose",
                height=320,
                showlegend=False,
                margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig, width="stretch")

        st.caption(
            f"Hill fits use {sum(hf.n_points for hf in eligible.values())} monotherapy "
            "data points from your uploaded CSV. "
            "DRI > 1 = combination needs less drug than monotherapy for the same effect."
        )


def render_result_visualization(active_results, res):
    """Renders the plot visualization for the active optimization result."""
    st.subheader("Visualize Result")

    model_to_plot_viz = None
    optimized_dosages_dict = None

    if active_results == "single":
        model_to_plot_viz = res['model_name']
        optimized_dosages_dict = dict(zip(st.session_state.independent_vars, res['dosages']))
    elif active_results == "advanced_tradeoff":
        top_5_weighted = res.get('top_5_weighted')
        if top_5_weighted:
            top_solution = top_5_weighted[0]
            dosages = top_solution.x
            model_choices = [res['model_1_name'], res['model_2_name']]
            formatted_viz_models = format_variable_options(model_choices)
            selected_viz_model_formatted = st.selectbox("Select model surface to visualize", options=formatted_viz_models, key="constrained_viz_model")
            model_to_plot_viz = selected_viz_model_formatted.split(":")[0]
            optimized_dosages_dict = dict(zip(st.session_state.independent_vars, dosages))
    elif active_results == "classic_multi":
        dosages = res['dosages']
        model_choices = [res['model_1_name'], res['model_2_name']]
        formatted_viz_models = format_variable_options(model_choices)
        selected_viz_model_formatted = st.selectbox("Select model surface to visualize", options=formatted_viz_models, key="classic_viz_model")
        model_to_plot_viz = selected_viz_model_formatted.split(":")[0]
        optimized_dosages_dict = dict(zip(st.session_state.independent_vars, dosages))

    if len(st.session_state.independent_vars) < 2:
        if model_to_plot_viz and optimized_dosages_dict:
            render_2d_result_plot(model_to_plot_viz, optimized_dosages_dict)
    else:
        if model_to_plot_viz and optimized_dosages_dict:
            render_3d_result_plot(model_to_plot_viz, optimized_dosages_dict)


def render_2d_result_plot(model_to_plot_viz, optimized_dosages_dict):
    """Renders the 2D visualization of an optimization result."""
    single_var = st.session_state.independent_vars[0]
    model_obj = st.session_state.wrapped_models[model_to_plot_viz]

    with st.expander("⚙️ Customize Plot Appearance"):
        st.write("**Optimized Point Marker**")
        marker_symbols = ['x', 'circle', 'diamond', 'square', 'cross']
        m_c1, m_c2, m_c3 = st.columns(3)
        opt_point_color_2d = m_c1.color_picker("Color", value="#D30303", key="opt_viz_2d_point_color")
        opt_point_symbol_2d = m_c2.selectbox("Symbol", options=marker_symbols, key="opt_viz_2d_point_symbol", index=0)
        opt_point_size_2d = m_c3.slider("Size", 8, 24, 15, key="opt_viz_2d_point_size")

    fig = plot_response_curve(
        dataframe=st.session_state.expanded_df,
        OLS_model=model_obj,
        independent_var=single_var,
        dependent_var=model_to_plot_viz,
        variable_descriptions=st.session_state.variable_descriptions,
        optimized_point=optimized_dosages_dict,
        opt_point_color=opt_point_color_2d,
        opt_point_symbol=opt_point_symbol_2d,
        opt_point_size=opt_point_size_2d
    )
    st.plotly_chart(fig, width="stretch")

def render_3d_result_plot(model_to_plot_viz, optimized_dosages_dict):
    """Renders the 3D visualization of an optimization result."""
    st.write(f"Plotting surface for **{model_to_plot_viz}** with the optimized point.")
    viz_c1, viz_c2 = st.columns(2)
    formatted_vars_viz = format_variable_options(st.session_state.independent_vars)

    x_var_viz_formatted = viz_c1.selectbox("X-axis variable", options=formatted_vars_viz, key="opt_viz_x")
    x_var_viz = x_var_viz_formatted.split(":")[0]

    y_options_viz_formatted = [v for v in formatted_vars_viz if not v.startswith(x_var_viz)]
    y_var_viz_formatted = viz_c2.selectbox("Y-axis variable", options=y_options_viz_formatted, key="opt_viz_y")
    y_var_viz = y_var_viz_formatted.split(":")[0]

    fixed_vars_for_plot = {k: v for k, v in optimized_dosages_dict.items() if k not in [x_var_viz, y_var_viz]}

    st.write("**Fixed Values (from optimization result):**")
    if not fixed_vars_for_plot:
        st.info("All variables are shown on the plot axes.")
    else:
        for var, val in fixed_vars_for_plot.items():
            st.markdown(f"- **{st.session_state.variable_descriptions.get(var, var)}**: `{val:.4f}`")

    model_to_plot_viz_2 = None
    colorscale_2_viz = 'Greys'
    optimized_point_2 = None
    point_color_2_viz = '#00FF00' # Green
    point_symbol_2_viz = 'cross'
    
    marker_symbols = ['x', 'circle', 'diamond', 'square', 'cross', 'circle-open', 'diamond-open']
    colorscale_options = ['Viridis', 'Plasma', 'Jet', 'Cividis', 'Hot', 'Cool', 'Blues', 'Greens', 'RdBu', 'Greys']

    with st.expander("📊 Overlay a comparison surface"):
        compare_in_results = st.checkbox("Plot a second model surface", key="compare_in_results_viz")
        if compare_in_results:
            all_model_keys = list(st.session_state.wrapped_models.keys())
            model_options_2_viz = [m for m in all_model_keys if m != model_to_plot_viz]
            if model_options_2_viz:
                c1_comp, c2_comp = st.columns([2, 1])
                with c1_comp:
                    formatted_model_options_2 = format_variable_options(model_options_2_viz)
                    selected_model_2_formatted = st.selectbox("Select Comparison Model", options=formatted_model_options_2, key="viz_model_2")
                    model_to_plot_viz_2 = selected_model_2_formatted.split(":")[0]
                with c2_comp:
                     colorscale_2_viz = st.selectbox("Comparison Color Scheme", options=colorscale_options, key="viz_colorscale_2", index=9)
                
                st.info(f"The comparison surface for **{model_to_plot_viz_2}** will also use the fixed values and optimal dosages determined by the primary optimization.")
                optimized_point_2 = optimized_dosages_dict # Use the same point for the second model
                
                st.write("**Optimal Point Marker (Comparison Model)**")
                m2_c1, m2_c2 = st.columns(2)
                point_color_2_viz = m2_c1.color_picker("Color", value="#00FF00", key="viz_point_color_2")
                point_symbol_2_viz = m2_c2.selectbox("Symbol", options=marker_symbols, key="viz_point_symbol_2", index=4)

            else:
                st.warning("No other models are available to compare.")

    with st.expander("⚙️ Customize Plot Appearance"):
        st.write("**Optimal Point Marker (Primary Model)**")
        p1_c1, p1_c2 = st.columns(2)
        point_color_1_viz = p1_c1.color_picker("Color", value="#FF0000", key="viz_point_color_1")
        point_symbol_1_viz = p1_c2.selectbox("Symbol", options=marker_symbols, key="viz_point_symbol_1", index=0)
        
        st.write("**Z-Axis Range**")
        enable_z_limit_opt = st.checkbox("Set Manual Z-Axis Range", value=False, key="opt_z_manual")
        z_range = None
        if enable_z_limit_opt:
            z_c1, z_c2 = st.columns(2)
            default_min = float(st.session_state.exp_df[model_to_plot_viz].min())
            default_max = float(st.session_state.exp_df[model_to_plot_viz].max())
            z_min = z_c1.number_input("Min Z Value", value=default_min, key="opt_z_min")
            z_max = z_c2.number_input("Max Z Value", value=default_max, key="opt_z_max")
            z_range = [z_min, z_max]

        st.write("---")
        st.write("**Surface Grid Lines**")
        g_c1, g_c2, g_c3 = st.columns(3)
        show_x_grid_viz = g_c1.checkbox("Show X-axis grid", value=True, key="viz_show_x_grid")
        show_y_grid_viz = g_c2.checkbox("Show Y-axis grid", value=True, key="viz_show_y_grid")
        show_surface_grid_viz = g_c3.checkbox("Show Surface Grid (Wireframe)", value=True, key="viz_show_surface_grid")

    plot_params_viz = {
        'x_var': x_var_viz, 'y_var': y_var_viz,
        'z_var_1': model_to_plot_viz,
        'fixed_vars_dict_1': fixed_vars_for_plot,
        'z_var_2': model_to_plot_viz_2,
        'fixed_vars_dict_2': fixed_vars_for_plot if model_to_plot_viz_2 else None,
        'variable_descriptions': st.session_state.variable_descriptions,
        'optimized_point': optimized_dosages_dict,
        'optimized_point_2': optimized_point_2,
        'show_actual_data': False,
        'colorscale_2': colorscale_2_viz,
        'show_x_grid': show_x_grid_viz,
        'show_y_grid': show_y_grid_viz,
        'show_surface_grid': show_surface_grid_viz,
        'point_symbol_1': point_symbol_1_viz,
        'point_color_1': point_color_1_viz,
        'point_symbol_2': point_symbol_2_viz,
        'point_color_2': point_color_2_viz,
        'z_range': z_range
    }
    display_surface_plot(plot_params_viz)
