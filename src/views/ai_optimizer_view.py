# src/views/ai_optimizer_view.py
import streamlit as st
import pandas as pd
import math
from datetime import datetime
from logic.plotting import plot_combination_ranking, plot_response_curve
from logic.optimization import run_bayesian_optimization, run_combination_optimization
from logic.data_processing import generate_optimization_report
from utils.ui_helpers import (format_variable_options, get_optimization_bounds_and_algo,
                              validate_bounds_for_ai, display_surface_plot)
from utils.state_management import clear_optimizer_results
def render(analysis_type=None, formatted_models=None):
    """Renders the AI (Bayesian) optimizer.

    When `analysis_type` / `formatted_models` are passed (from the merged
    Optimization tab) the standalone header and analysis-type radio are skipped.
    """
    if analysis_type is None:
        st.subheader("🤖 AI Optimizer (Bayesian Optimization)")
        st.info("""
    This advanced optimizer uses a smart search strategy (Bayesian Optimization) to efficiently find the best possible outcomes.
    - **Optimize All Variables:** Finds the single best combination of all variables to meet your goal.
    - **Combination Analysis:** Systematically tests combinations of 2 or 3 variables to rank which groups of variables are the most impactful.
    """)
        analysis_type = st.radio(
            "Select Analysis Type",
            ("Optimize All Variables", "Combination Analysis"),
            key="bopt_analysis_type",
            horizontal=True,
            on_change=clear_optimizer_results
        )
        st.write("---")

    if formatted_models is None:
        formatted_models = format_variable_options(st.session_state.wrapped_models.keys())
    if not formatted_models:
        st.info("Run an analysis from the 'Project Library' to use the AI Optimizer.")
        return

    selected_model_formatted_bopt = st.selectbox(
        "Select model to optimize",
        options=formatted_models,
        key="bopt_model",
        on_change=clear_optimizer_results
    )
    model_to_optimize_bopt = selected_model_formatted_bopt.split(":")[0]
    
    bopt_type = st.radio(
        "Optimization Goal",
        ("Minimize/Maximize", "Target a Specific Value"),
        key="bopt_type",
        horizontal=True,
        on_change=clear_optimizer_results
    )

    target_value_bopt = None
    goal_bopt = "Minimize"
    if bopt_type == "Minimize/Maximize":
        goal_bopt = st.radio(
            "Goal", ("Minimize", "Maximize"), key="bopt_goal",
            horizontal=True, on_change=clear_optimizer_results
        )
    else:
        target_value_bopt = st.number_input(
            "Target Value", value=0.0, format="%.4f",
            key="bopt_target", on_change=clear_optimizer_results
        )

    st.write("---")
    discrete_vars_bopt = st.multiselect(
        "Select any variables that are discrete",
        options=st.session_state.independent_vars,
        default=st.session_state.detected_binary_vars,
        key="bopt_discrete_vars",
        help="The optimizer will only choose from the specific integer values found in the data for these variables."
    )

    bounds_bopt = None
    combo_size = None
    outcome_min_bopt = None
    if analysis_type == "Optimize All Variables":
        bounds_bopt = get_optimization_bounds_and_algo("bopt")
    else: # Combination Analysis
        st.write("**Combination Settings**")
        combo_size = st.number_input(
            "Number of variables per combination (2 or 3)",
            min_value=2, max_value=3, value=2, step=1, key="combo_size"
        )
        outcome_min_bopt = st.number_input(
            "Minimum acceptable outcome",
            value=0.0, format="%.4f", key="combo_outcome_min",
            help="Combinations that produce an outcome below this value will be penalised. Set to 0 to prevent physically impossible negative results (e.g. cell viability)."
        )
        st.info(f"The optimizer will test all possible combinations of **{combo_size}** variables from your list of independent variables.")

    with st.expander("⚙️ Advanced AI Optimizer Settings"):
        n_calls = st.number_input(
            "Number of Iterations per Combination (n_calls)", min_value=1, value=25,
													  
            key="bopt_n_calls", help="More calls can lead to better results but take longer."
        )
        n_initial_points = st.number_input(
            "Number of Initial Random Points", min_value=1, value=10,
														
            key="bopt_n_initial", help="The number of random points to sample before the intelligent search begins."
        )

    if st.button("Run AI Optimization", type="primary"):
        run_ai_optimization(
            analysis_type, model_to_optimize_bopt, goal_bopt, target_value_bopt,
            discrete_vars_bopt, bounds_bopt, combo_size, n_calls, n_initial_points,
            outcome_min_bopt
        )
    
    if st.session_state.get("bayesian_opt_results"):
        render_bayesian_opt_results(model_to_optimize_bopt, target_value_bopt)
    
    if st.session_state.get("bayesian_combo_results"):
        render_bayesian_combo_results()

def run_ai_optimization(bopt_analysis_type, model_name, goal, target_value,
                        discrete_vars, bounds, combo_size, n_calls, n_initial_points,
                        outcome_min=None):
    """Handles the logic for triggering the correct AI optimization function."""
    clear_optimizer_results() # Clear previous optimization results
    selected_model_bopt = st.session_state.wrapped_models[model_name]
    
    if bopt_analysis_type == "Optimize All Variables":
        if not validate_bounds_for_ai(bounds, st.session_state.independent_vars, st.session_state.variable_descriptions):
            st.stop()
        with st.spinner("Running Bayesian Optimization... This may take a moment."):
            try:
                discrete_vars_dict = {
                    var: st.session_state.unique_variable_values[var] for var in discrete_vars
                }
                results = run_bayesian_optimization(
                    OLS_model=selected_model_bopt, all_alphabet_vars=st.session_state.independent_vars,
                    bounds=bounds, goal=goal, n_calls=n_calls, n_initial_points=n_initial_points,
                    variable_descriptions=st.session_state.variable_descriptions,
                    discrete_vars=discrete_vars_dict, target_value=target_value
                )
                st.session_state.bayesian_opt_results = results
                st.session_state.bayesian_opt_report_data = create_bopt_report_data(
                    model_name, selected_model_bopt, goal, target_value, n_calls, n_initial_points, bounds, results
                )
                st.success("AI Optimization Successful!")
            except Exception as e:
                st.error(f"An error occurred during Bayesian Optimization: {e}")

    else: # Combination Analysis
        num_vars = len(st.session_state.independent_vars)
        num_combinations = math.comb(num_vars, combo_size)
        st.warning(f"Starting parallel analysis of **{num_combinations}** combinations. Your CPU usage will be high. This may take a significant amount of time.")
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Initializing parallel worker processes...")

        try:
            results = run_combination_optimization(
                OLS_model=selected_model_bopt, all_independent_vars=st.session_state.independent_vars,
                variable_stats=st.session_state.variable_stats, variable_descriptions=st.session_state.variable_descriptions,
                combo_size=combo_size, goal=goal, target_value=target_value, n_calls=n_calls,
                n_initial_points=n_initial_points, progress_bar=progress_bar, status_text=status_text,
                outcome_min=outcome_min
            )
            st.session_state.bayesian_combo_results = results
            status_text.success("Combination Analysis Complete!")
        except Exception as e:
            st.error(f"An error occurred during Combination Analysis: {e}")

def create_bopt_report_data(model_name, model, goal, target_value, n_calls, n_initial, bounds, results):
    """Helper to assemble data for the optimization report."""
    
    formula = "N/A"
    if hasattr(model, 'formula'):
        formula = model.formula

    params = None
    if hasattr(model, 'params'):
        params = model.params

    return {
        "optimization_type": "AI - Bayesian",
        "model_name": model_name,
        "model_formula": formula,
        "model_params": params,
        "settings": {
            "Goal": "Target" if target_value is not None else goal,
            "Target Value": target_value if target_value is not None else "N/A",
            "Iterations": n_calls,
            "Initial Points": n_initial,
            "Bounds": dict(zip(st.session_state.independent_vars, bounds)),
        },
        "results": {
            "Status": "Success",
            "Final Outcome": results['outcome'],
            "Optimal Dosages": dict(zip(st.session_state.independent_vars, results['dosages']))
        }
    }

def render_bayesian_opt_results(model_to_optimize_bopt, target_value_bopt):
    """Displays the results for the 'Optimize All Variables' analysis."""
    st.write("---")
    st.header("AI Optimization Result")

    report_data_bopt = st.session_state.get("bayesian_opt_report_data")
    col1_b, col2_b = st.columns([3, 1])
    with col2_b:
        if report_data_bopt:
            report_bytes_bopt = generate_optimization_report(report_data_bopt, st.session_state.variable_descriptions)
            st.download_button(
                label="📄 Download Report", data=report_bytes_bopt,
                file_name=f"Report_AI_Bayesian_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="bopt_report_download"
            )
    with col1_b:
        if st.button("Clear AI Results"):
            clear_optimizer_results()
            st.rerun()

    res_bopt = st.session_state.bayesian_opt_results
    descriptions = st.session_state.variable_descriptions
    model_desc_bopt = descriptions.get(model_to_optimize_bopt, model_to_optimize_bopt)

    if target_value_bopt is not None:
        st.metric(label=f"Closest Value to Target ({target_value_bopt}) for {model_desc_bopt}", value=f"{res_bopt['outcome']:.4f}")
    else:
        st.metric(label=f"Optimal {model_desc_bopt} Value", value=f"{res_bopt['outcome']:.4f}")

    st.write("**Optimal Dosages:**")
    descriptive_vars_bopt = [descriptions.get(var, var) for var in st.session_state.independent_vars]
    dosages_df_bopt = pd.DataFrame({'Variable': descriptive_vars_bopt, 'Optimal Value': res_bopt['dosages']})
    st.dataframe(dosages_df_bopt, width="stretch") 

    st.write("---")
    st.subheader("Analysis Plots")
    st.write("**Convergence Plot:** This plot shows the best value found so far at each iteration. A flattening curve suggests the optimizer has converged to a solution.")
    st.pyplot(res_bopt['convergence_plot'])

    st.write("**Objective Plot:** This plot shows how the objective function is expected to change with each variable, holding others at their optimal values. It helps understand the sensitivity and influence of each parameter.")
    st.pyplot(res_bopt['objective_plot'])

    st.write("---")
    st.subheader("Visualize Result")
    optimized_dosages_dict_bopt = dict(zip(st.session_state.independent_vars, res_bopt['dosages']))
    if len(st.session_state.independent_vars) < 2:
        render_bopt_2d_plot(model_to_optimize_bopt, optimized_dosages_dict_bopt)
    else:
        render_bopt_3d_plot(model_to_optimize_bopt, optimized_dosages_dict_bopt)

def render_bopt_2d_plot(model_name, dosages_dict):
    """Renders the 2D plot for AI optimization results."""
    single_var_bopt = st.session_state.independent_vars[0]
    model_obj_bopt = st.session_state.wrapped_models[model_name]
    fig_bopt_2d = plot_response_curve(
        dataframe=st.session_state.expanded_df, OLS_model=model_obj_bopt,
        independent_var=single_var_bopt, dependent_var=model_name,
        variable_descriptions=st.session_state.variable_descriptions,
        optimized_point=dosages_dict, optimized_point_name='AI Optimized Point'
    )
    st.plotly_chart(fig_bopt_2d, width="stretch")

def render_bopt_3d_plot(model_name, dosages_dict):
    """Renders the 3D plot for AI optimization results."""
    st.write(f"Plotting surface for **{model_name}** with the AI-optimized point.")
    viz_c1, viz_c2 = st.columns(2)
    formatted_vars_viz_bopt = format_variable_options(st.session_state.independent_vars)
    x_var_viz_formatted_bopt = viz_c1.selectbox("X-axis variable", options=formatted_vars_viz_bopt, key="bopt_viz_x")
    x_var_viz_bopt = x_var_viz_formatted_bopt.split(":")[0]
    y_options_viz_formatted_bopt = [v for v in formatted_vars_viz_bopt if not v.startswith(x_var_viz_bopt)]
    y_var_viz_formatted_bopt = viz_c2.selectbox("Y-axis variable", options=y_options_viz_formatted_bopt, key="bopt_viz_y")
    y_var_viz_bopt = y_var_viz_formatted_bopt.split(":")[0]
    fixed_vars_for_plot_bopt = {k: v for k, v in dosages_dict.items() if k not in [x_var_viz_bopt, y_var_viz_bopt]}

    st.write("**Fixed Values (from AI optimization result):**")
    if not fixed_vars_for_plot_bopt:
        st.info("All variables are shown on the plot axes.")
    else:
        for var, val in fixed_vars_for_plot_bopt.items():
            st.markdown(f"- **{st.session_state.variable_descriptions.get(var, var)}**: `{val:.4f}`")

    with st.expander("🎨 Customize Plot Appearance"):
        st.write("**Z-Axis Range**")
        enable_z_limit_bopt = st.checkbox("Set Manual Z-Axis Range", value=False, key="bopt_z_manual")
        z_range = None
        if enable_z_limit_bopt:
            z_c1, z_c2 = st.columns(2)
            default_min = float(st.session_state.exp_df[model_name].min())
            default_max = float(st.session_state.exp_df[model_name].max())
            z_min = z_c1.number_input("Min Z Value", value=default_min, key="bopt_z_min")
            z_max = z_c2.number_input("Max Z Value", value=default_max, key="bopt_z_max")
            z_range = [z_min, z_max]

        st.write("---")
        st.write("**Surface Grid Lines**")
        g_c1, g_c2, g_c3 = st.columns(3)
        show_x_grid_bopt = g_c1.checkbox("Show X-axis grid", value=True, key="bopt_show_x_grid")
        show_y_grid_bopt = g_c2.checkbox("Show Y-axis grid", value=True, key="bopt_show_y_grid")
        show_surface_grid_bopt = g_c3.checkbox("Show Surface Grid (Wireframe)", value=True, key="bopt_show_surface_grid")

    plot_params_viz_bopt = {
        'x_var': x_var_viz_bopt, 'y_var': y_var_viz_bopt,
        'z_var_1': model_name,
        'fixed_vars_dict_1': fixed_vars_for_plot_bopt,
        'variable_descriptions': st.session_state.variable_descriptions,
        'optimized_point': dosages_dict,
        'show_actual_data': False,
        'z_range': z_range,
        'show_x_grid': show_x_grid_bopt,
        'show_y_grid': show_y_grid_bopt,
        'show_surface_grid': show_surface_grid_bopt
    }
    display_surface_plot(plot_params_viz_bopt)

def render_bayesian_combo_results():
    """Displays the results for the 'Combination Analysis'."""
    st.write("---")
    st.header("Combination Analysis Result")
    if st.button("Clear AI Results"):
        clear_optimizer_results()
        st.rerun()

    res_combo = st.session_state.bayesian_combo_results
    sorted_results = res_combo.get('sorted_results', [])

    if not sorted_results:
        st.warning("No valid combinations were found.")
        return

    st.subheader("🏆 Top Performing Combinations")
    best_combo_info = sorted_results[0]
    with st.container(border=True):
        st.markdown("🥇 **Best Combination Found**")
        st.metric(label="Best Outcome Achieved", value=f"{best_combo_info['outcome']:.4f}")
        st.write("**Best Performing Variables:**")
        st.code(" & ".join(best_combo_info['descriptive_combination']))
        st.write("**Optimal Dosages for this Combination:**")
        dosages_df_combo = pd.DataFrame(best_combo_info['descriptive_dosages'].items(), columns=['Variable', 'Optimal Value'])
        st.dataframe(dosages_df_combo, width="stretch") 

    if len(sorted_results) > 1:
        st.write("---")
        cols = st.columns(2)
        with cols[0]:
            with st.container(border=True):
                st.markdown("🥈 **Second Best Combination**")
                second_combo_info = sorted_results[1]
                st.metric(label="Outcome", value=f"{second_combo_info['outcome']:.4f}")
                st.code(" & ".join(second_combo_info['descriptive_combination']))
                dosages_df_second = pd.DataFrame(second_combo_info['descriptive_dosages'].items(), columns=['Variable', 'Optimal Value'])
                st.dataframe(dosages_df_second, width="stretch")

        if len(sorted_results) > 2:
            with cols[1]:
                 with st.container(border=True):
                    st.markdown("🥉 **Third Best Combination**")
                    third_combo_info = sorted_results[2]
                    st.metric(label="Outcome", value=f"{third_combo_info['outcome']:.4f}")
                    st.code(" & ".join(third_combo_info['descriptive_combination']))
                    dosages_df_third = pd.DataFrame(third_combo_info['descriptive_dosages'].items(), columns=['Variable', 'Optimal Value'])
                    st.dataframe(dosages_df_third, width="stretch")

    st.write("*(Other variables were held at the midpoint of their range during this analysis.)*")

    st.write("---")
    st.subheader("Ranking of All Combinations")
    st.info("This bar chart ranks the best possible outcome achieved by every combination tested. The best performers are at the top.")
    ranking_df = res_combo.get('ranking_df')
    if ranking_df is not None:
        goal = "Target" if st.session_state.get("bopt_target") is not None else st.session_state.get("bopt_goal", "Maximize")
        ranking_plot_fig = plot_combination_ranking(ranking_df, goal)
        st.pyplot(ranking_plot_fig)
