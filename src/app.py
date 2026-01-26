# src/app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import multiprocessing
import json
import sys
if not hasattr(np, 'int'):
    np.int = int
from views import library_view, plotting_view, optimizer_view, ai_optimizer_view, synergy_view, diagnostics_view
from utils import state_management
from logic.models import RandomForestWrapper

def display_sidebar_info():
    """Displays information about the loaded project in the sidebar."""
    st.sidebar.header("Project Status")

    if not st.session_state.get('processed_file'):
        st.sidebar.info("No project loaded. Upload a new CSV or load a project from the Project Library.")
    else:
        st.sidebar.success(f"✅ Loaded: **{st.session_state.processed_file}**")
        with st.sidebar.expander("Loaded Data Info", expanded=False):
            st.write(f"**Independent Vars:** `{len(st.session_state.get('independent_vars', []))}`")
            st.write(f"**Dependent Vars:** `{len(st.session_state.get('dependent_vars', []))}`")

    st.sidebar.write("---")
    
    active_run = st.session_state.get('active_analysis_run')
    if not active_run and st.session_state.get('processed_file'):
        st.sidebar.warning("Project data is loaded. Go to the Library to run or load an analysis.")
    
    if active_run:
        st.sidebar.subheader(f"📊 Active Analysis: {active_run}")
        desc_map = st.session_state.get('variable_descriptions', {})
        library = library_view.load_library()
        project_data = library.get(st.session_state.processed_file, {})
        run_data = project_data.get('analysis_runs', {}).get(active_run, {})
        model_type = run_data.get('model_type', 'Unknown')
        st.sidebar.markdown(f"**Method:** `{model_type}`")

        for model_name, wrapped_model in st.session_state.get('wrapped_models', {}).items():
            description = desc_map.get(model_name, model_name)
            with st.sidebar.expander(f"**{model_name}**: ({description})"):
                summary = wrapped_model.get_summary()
                st.code(summary)


def main():
    """
    Main function to run the Streamlit application UI and logic.
    """
    st.set_page_config(page_title="AI-PRS Analysis Tool", layout="wide")
    st.title("🧠 Response Surface Analysis Tool")

    state_management.initialize_session_state()
    display_sidebar_info()

    st.header("Analysis & Visualization")
    
    analysis_is_done = st.session_state.get('analysis_done', False)
    
    # --- UPDATED TAB LIST (Removed 'Actual vs. Predicted') ---
    tab_list = [
        "📚 Project Library", 
        "🧊 Plotting Tools", 
        "🔍 Diagnostics", 
        "🎯 Optimizer",
        "🤖 AI Optimizer", 
        "🤝 Synergy Analysis", 
        "💾 Session State"
    ]
    
    # Unpack 7 tabs (previously 8)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_list)

    with tab1:
        library_view.render()

    with tab2:
        if analysis_is_done:
            plotting_view.render()
        else:
            st.info("Load and analyze a project from the 'Project Library' to use the plotting tools.")

    # Diagnostics Tab (Now covers the old Evaluation features + more)
    with tab3:
        if analysis_is_done:
            diagnostics_view.render()
        else:
            st.info("Load and analyze a project from the 'Project Library' to view OLS diagnostics.")

    with tab4:
        if analysis_is_done:
            active_model_is_rf = False
            if st.session_state.get('wrapped_models'):
                # Check if any of the active models is a Random Forest model
                active_model_is_rf = any(isinstance(model, RandomForestWrapper) for model in st.session_state.wrapped_models.values())
            
            if active_model_is_rf:
                st.warning("The standard Optimizer tab is not compatible with Random Forest models. Please use the AI Optimizer tab for optimization.", icon="⚠️")
            else:
                optimizer_view.render()
        else:
            st.info("Load and analyze a project from the 'Project Library' to use the optimizer.")

    with tab5:
        if analysis_is_done:
            ai_optimizer_view.render()
        else:
            st.info("Load and analyze a project from the 'Project Library' to use the AI optimizer.")
            
    with tab6:
        if analysis_is_done:
            synergy_view.render()
        else:
            st.info("Load and analyze a project from the 'Project Library' to perform synergy analysis.")
            
    # Session State Tab (Renumbered to 7)
    with tab7:
        render_session_state_tab()

def render_session_state_tab():
    """Renders the content for the Session State tab, including the password unlock."""
    st.subheader("Current Session State (for debugging)")
    
    with st.expander("Loaded DataFrames"):
        st.write("#### Original Data (`exp_df`)")
        st.dataframe(st.session_state.get("exp_df"))
        st.write("#### Expanded Data with Polynomial Terms (`expanded_df`)")
        st.dataframe(st.session_state.get("expanded_df"))

    with st.expander("Wrapped Model Objects (`wrapped_models`)"):
        if st.session_state.get('wrapped_models'):
            for model_name, model_obj in st.session_state.wrapped_models.items():
                st.write(f"##### Model: {model_name}")
                st.code(model_obj.get_summary())
        else:
            st.info("No analysis has been run yet.")
    
    with st.expander("Standard Optimizer Results"):
        st.write("**Single-Objective Result:**", st.session_state.get("single_opt_results", "Not run yet."))
        st.write("**Classic Multi-Objective Result:**", st.session_state.get("classic_multi_opt_results", "Not run yet."))
        st.write("**Weighted Score Multi-Objective Result:**", st.session_state.get("advanced_tradeoff_results", "Not run yet."))

    if st.session_state.get('experimental_unlocked', False):
        with st.expander("AI Optimizer Results"):
            st.write("**Bayesian (Optimize All) Result:**")
            if st.session_state.get("bayesian_opt_results"):
                display_dict = {k: v for k, v in st.session_state.bayesian_opt_results.items() if k not in ['convergence_plot', 'objective_plot', 'raw_result']}
                st.json(display_dict)
            else:
                st.info("Not run yet.")

            st.write("**Bayesian (Combination) Result:**")
            if st.session_state.get("bayesian_combo_results"):
                display_dict = {k: v for k, v in st.session_state.bayesian_combo_results.items() if k != 'ranking_plot'}
                st.json(display_dict)
            else:
                st.info("Not run yet.")

        with st.expander("Synergy Results"):
             st.write("**Synergy Matrix:**", st.session_state.get("synergy_matrix", "Not run yet."))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
