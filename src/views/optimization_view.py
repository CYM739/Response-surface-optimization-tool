# src/views/optimization_view.py
"""Unified Optimization tab.

Merges the standard (SciPy / gradient) and AI (Bayesian) optimizers behind a
single Engine -> Strategy selector. All optimization *logic* is reused from
optimizer_view and ai_optimizer_view — this module only routes.
"""
import streamlit as st
from utils.ui_helpers import format_variable_options
from utils.state_management import clear_optimizer_results
from views import optimizer_view, ai_optimizer_view


def render():
    st.subheader("🎯 Optimization")

    if not st.session_state.get('wrapped_models'):
        st.info("Please analyze a project in the 'Project Library' before using the optimizer.")
        return

    formatted_models = format_variable_options(st.session_state.wrapped_models.keys())

    engine = st.radio(
        "Optimization engine",
        ["SciPy (gradient)", "Bayesian (AI)"],
        horizontal=True,
        key="opt_engine",
        on_change=clear_optimizer_results,
        help="**SciPy**: fast gradient search on a smooth surface. "
             "**Bayesian (AI)**: gradient-free smart search — works for any model "
             "(including non-smooth Random Forest) and for combination screening.",
    )

    if engine == "SciPy (gradient)":
        # Gradient optimizer needs a smooth surface (capability flag).
        incompatible = [name for name, m in st.session_state.wrapped_models.items()
                        if not getattr(m, 'SUPPORTS_GRADIENT_OPT', True)]
        if incompatible:
            st.warning(
                "The gradient (SciPy) optimizer needs a smooth surface. "
                f"Incompatible model(s): {', '.join(incompatible)}. "
                "Switch **Engine** to **Bayesian (AI)** to optimize this model.",
                icon="⚠️",
            )
            return

        strategies = ["Single-objective", "Multi-objective: Classic two-stage"]
        if st.session_state.get('experimental_unlocked', False):
            strategies.append("Multi-objective: Weighted score")

        strategy = st.selectbox(
            "Strategy", strategies, key="scipy_strategy",
            on_change=clear_optimizer_results,
        )
        if strategy == "Single-objective":
            optimizer_view.render_single_objective_optimizer(formatted_models)
        elif "Classic" in strategy:
            optimizer_view.render_classic_two_stage_ui(formatted_models)
        else:
            optimizer_view.render_weighted_score_ui(formatted_models)

        optimizer_view.render_optimizer_results()

    else:  # Bayesian (AI)
        strategy = st.selectbox(
            "Strategy", ["Optimize All Variables", "Combination Analysis"],
            key="bayes_strategy", on_change=clear_optimizer_results,
        )
        ai_optimizer_view.render(analysis_type=strategy, formatted_models=formatted_models)
