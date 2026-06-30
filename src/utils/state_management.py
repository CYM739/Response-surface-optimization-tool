# src/utils/state_management.py
import streamlit as st

_OPTIMIZER_RESULT_KEYS = [
    'single_opt_results', 'multi_opt_results', 'advanced_tradeoff_results',
    'classic_multi_opt_results', 'bayesian_opt_results', 'bayesian_combo_results',
    'single_opt_report_data', 'bayesian_opt_report_data',
]

def initialize_session_state():
    """
    Initializes all required keys in Streamlit's session state on the first
    run of the application.
    """
    if 'analysis_done' not in st.session_state:
        st.session_state.exp_df = None
        st.session_state.expanded_df = None
        st.session_state.all_vars = []
        st.session_state.independent_vars = []
        st.session_state.dependent_vars = []
        st.session_state.variable_stats = {}
        st.session_state.unique_variable_values = {}
        st.session_state.variable_descriptions = {}
        st.session_state.wrapped_models = {}
        st.session_state.analysis_done = False
        st.session_state.processed_file = None
        st.session_state.active_analysis_run = None
        st.session_state.detected_binary_vars = []

        # Optimizer States
        st.session_state.single_opt_results = None
        st.session_state.multi_opt_results = None  # ADDED FOR EDUCATION MULTI-OBJECTIVE
        st.session_state.advanced_tradeoff_results = None # For weighted-score
        st.session_state.classic_multi_opt_results = None # For classic two-stage
        st.session_state.bayesian_opt_results = None
        st.session_state.bayesian_combo_results = None
        
        # Synergy States
        st.session_state.synergy_matrix = None
        st.session_state.synergy_drugs = None
        st.session_state.synergy_model_name = None

        # Report Data States
        st.session_state.single_opt_report_data = None
        st.session_state.bayesian_opt_report_data = None
        
        # Hill Fit / DRI States
        st.session_state.hill_fits = None   # {dep_var: {drug: HillFit}} — set after analysis

        # Set experimental features to be unlocked by default
        st.session_state.experimental_unlocked = True

def reset_state():
    """
    Resets all application-specific keys to their initial state.
    """
    # Keep the unlock status across project loads
    unlocked_status = st.session_state.get('experimental_unlocked', False)

    keys_to_reset = [
        'exp_df', 'expanded_df', 'all_vars', 'independent_vars', 'dependent_vars',
        'variable_stats', 'unique_variable_values', 'variable_descriptions',
        'wrapped_models', 'processed_file', 'active_analysis_run',
        'detected_binary_vars', 'analysis_done'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
            
    initialize_session_state() # Re-initialize to default values
    st.session_state.experimental_unlocked = unlocked_status # Restore unlock status

    # Clear results from all tools
    clear_optimizer_results()
    clear_synergy_results()

def clear_synergy_results():
    """Clears synergy-related results from the session state."""
    st.session_state.synergy_matrix = None
    st.session_state.synergy_drugs = None
    st.session_state.synergy_model_name = None

def clear_optimizer_results():
    """
    A callback function to reset ALL optimization results in the session state.
    This is crucial to prevent state conflicts between different optimizers.
    """
    for key in _OPTIMIZER_RESULT_KEYS:
        st.session_state[key] = None