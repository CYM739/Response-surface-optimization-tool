# src/utils/ui_helpers.py
import streamlit as st
import warnings
from scipy.optimize import minimize, basinhopping, shgo
from logic.plotting import plot_response_surface

def format_variable_options(variables):
    """Creates user-friendly labels for select boxes, combining variable names and descriptions."""
    descriptions = st.session_state.variable_descriptions
    return [f"{var}: ({descriptions.get(var, 'No description')})" for var in variables]

def get_optimization_bounds_and_algo(key_prefix, default_algo='SHGO (Global)'):
    """
    Encapsulates the UI elements for setting optimization bounds and choosing a SciPy algorithm.
    This avoids code duplication across different optimizer tabs.
    """
    st.write("Define dosage bounds:")
    bounds = []
    for var in st.session_state.independent_vars:
        c1, c2 = st.columns(2)
        _ , second_min_v, max_v = st.session_state.variable_stats[var]
        desc = st.session_state.variable_descriptions.get(var, 'No description')
        min_bound = c1.number_input(f"Min bound for {var} ({desc})", value=float(second_min_v), key=f"{key_prefix}_min_{var}")
        max_bound = c2.number_input(f"Max bound for {var} ({desc})", value=float(max_v), key=f"{key_prefix}_max_{var}")
        bounds.append((min_bound, max_bound))

    if key_prefix.startswith("bopt"):
        return bounds

    start_points = [b[0] for b in bounds]
    st.write("---")
    with st.expander("⚙️ Advanced Optimizer Settings"):
        optimizer_options = ['SHGO (Global)', 'Basinhopping (Global)', 'SLSQP (Local)']
        try:
            default_index = optimizer_options.index(default_algo)
        except ValueError:
            default_index = optimizer_options.index('SHGO (Global)') # Fallback
            
        algorithm = st.selectbox(
            "Select Optimizer Algorithm",
            options=optimizer_options,
            index=default_index,
            key=f"{key_prefix}_algo",
            help="Global optimizers are more thorough but slower. SLSQP is fast but can get stuck in local minima."
        )
        algo_params = {}
        if algorithm.startswith('Basinhopping'):
            algo_params['niter'] = st.number_input("Number of Iterations (niter)", min_value=1, value=50, key=f"{key_prefix}_bh_niter")
        elif algorithm.startswith('SHGO'):
            algo_params['shgo_n'] = st.number_input("Sampling Points (n)", min_value=10, value=100, key=f"{key_prefix}_shgo_n")
            algo_params['shgo_iters'] = st.number_input("Local Search Iterations (iters)", min_value=1, value=3, key=f"{key_prefix}_shgo_iters")
            
    return bounds, start_points, algorithm, algo_params

def display_surface_plot(plot_params, plot_config=None):
    """A helper function to centralize the logic for rendering 3D surface plots."""
    if plot_config is None:
        plot_config = {}
    
    model_key_1 = plot_params.get('z_var_1')
    model_key_2 = plot_params.get('z_var_2')

    if not model_key_1 or model_key_1 not in st.session_state.wrapped_models:
        st.error(f"Could not find the specified primary model: {model_key_1}")
        return
    
    selected_model_1 = st.session_state.wrapped_models[model_key_1]
    selected_model_2 = None
    
    if model_key_2:
        if model_key_2 not in st.session_state.wrapped_models:
            st.error(f"Could not find the specified comparison model: {model_key_2}")
            return
        selected_model_2 = st.session_state.wrapped_models[model_key_2]
        
    fig = plot_response_surface(
        dataframe=st.session_state.expanded_df,
        OLS_model_1=selected_model_1, 
        OLS_model_2=selected_model_2,
        all_alphabet_vars=st.session_state.independent_vars,
        **plot_params
    )
    st.plotly_chart(fig, width="stretch", config=plot_config)

def validate_bounds_for_ai(bounds, independent_vars, variable_descriptions):
    """
    Validates that for every variable, the minimum bound is strictly less than the maximum bound.
    The Bayesian optimizer requires a valid search range (min < max) to explore effectively.
    If validation fails, it displays a user-friendly error message.
    """
    invalid_vars = []
    for i, bound in enumerate(bounds):
        if isinstance(bound, tuple):
            min_b, max_b = bound
            if min_b >= max_b:
                var_name = independent_vars[i]
                descriptive_name = variable_descriptions.get(var_name, var_name)
                invalid_vars.append(f"'{descriptive_name}'")
    
    if invalid_vars:
        error_message = (
            f"**AI Optimization cannot start.**\n\n"
            f"The following variable(s) have a minimum bound that is greater than or equal to the maximum bound: "
            f"**{', '.join(invalid_vars)}**.\n\n"
            f"The AI optimizer requires a valid search range for every variable. Please adjust the bounds and try again."
        )
        st.error(error_message, icon="🚨")
        return False
        
    return True