"""
Tier 1: headless control of the Streamlit app via streamlit.testing.v1.AppTest.

Boots `src/app.py` in an in-process Streamlit runtime (no browser, no port
8501), seeds st.session_state to mimic a loaded+analyzed project, then
exercises the Optimizer tab the same way a user would.

Fast enough that an AI agent can iterate many times per minute.
"""
import copy

import pytest
from streamlit.testing.v1 import AppTest

from logic.helpers import _add_polynomial_terms


APP_SCRIPT = "src/app.py"
BOOT_TIMEOUT = 60  # seconds — first boot imports torch-heavy modules transitively


# ---------------------------------------------------------------------------
# Smoke tests — no seeded state
# ---------------------------------------------------------------------------

def test_app_boots_without_exception():
    """Cold boot of the app must not raise. Catches syntax errors, bad imports,
    and state-init regressions before anything else."""
    at = AppTest.from_file(APP_SCRIPT, default_timeout=BOOT_TIMEOUT)
    at.run()
    assert not at.exception, f"App raised on cold boot: {at.exception}"


def test_title_and_tabs_render_on_cold_boot():
    at = AppTest.from_file(APP_SCRIPT, default_timeout=BOOT_TIMEOUT)
    at.run()

    titles = [t.value for t in at.title]
    assert any("Response Surface" in t for t in titles)

    # The app declares 10 tabs in app.py
    assert len(at.tabs) == 10


def test_optimizer_tab_guards_when_no_model_loaded():
    """With no project loaded, the Optimizer tab should show the info banner
    telling the user to analyze a project first — not crash."""
    at = AppTest.from_file(APP_SCRIPT, default_timeout=BOOT_TIMEOUT)
    at.run()
    assert not at.exception

    info_messages = [el.value for el in at.info]
    assert any(
        "Project Library" in msg and "optimizer" in msg.lower()
        for msg in info_messages
    ), f"Expected optimizer guard message. Got: {info_messages}"


# ---------------------------------------------------------------------------
# End-to-end: seed session_state with a trained model, drive the optimizer
# ---------------------------------------------------------------------------

def _seed_analyzed_project(at: AppTest, analyzed_demo: dict, trained_ols: dict):
    """Populate AppTest.session_state with the same keys library_view sets
    after 'Load Project Data' + 'Run and Save Analysis'. Lets us skip the UI
    dance for the upload/analyze flow and focus the test on the optimizer."""
    dep_var = trained_ols["dep_var"]
    wrapper = trained_ols["wrapper"]

    expanded = analyzed_demo["exp_df"].copy()
    _add_polynomial_terms(expanded, analyzed_demo["independent_vars"])

    # analysis_done=True prevents state_management.initialize_session_state()
    # from wiping our seeded values on first script run.
    at.session_state["analysis_done"] = True
    at.session_state["exp_df"] = analyzed_demo["exp_df"]
    at.session_state["expanded_df"] = expanded
    at.session_state["all_vars"] = analyzed_demo["all_vars"]
    at.session_state["independent_vars"] = analyzed_demo["independent_vars"]
    at.session_state["dependent_vars"] = analyzed_demo["dependent_vars"]
    at.session_state["variable_stats"] = analyzed_demo["variable_stats"]
    at.session_state["unique_variable_values"] = analyzed_demo["unique_variable_values"]
    at.session_state["variable_descriptions"] = analyzed_demo["variable_descriptions"]
    at.session_state["detected_binary_vars"] = analyzed_demo["detected_binary_vars"]
    at.session_state["wrapped_models"] = {dep_var: wrapper}
    at.session_state["processed_file"] = "demo_3drug_dri_fixture"
    at.session_state["active_analysis_run"] = "fixture_run"
    at.session_state["data_source_type"] = "csv"
    at.session_state["experimental_unlocked"] = True
    at.session_state["hill_fits"] = None

    # Optimizer result slots that views read unconditionally
    for key in [
        "single_opt_results", "multi_opt_results", "advanced_tradeoff_results",
        "classic_multi_opt_results", "bayesian_opt_results", "bayesian_combo_results",
        "single_opt_report_data", "bayesian_opt_report_data",
        "synergy_matrix", "synergy_drugs", "synergy_model_name",
        "braid_lookup_result", "braid_lookup_query", "braid_opt_result",
    ]:
        at.session_state[key] = None


def test_app_renders_with_seeded_project(analyzed_demo, trained_ols):
    """With a project pre-seeded, the Optimizer tab should now show the radio
    button (Single vs Multi-Objective) instead of the guard banner."""
    at = AppTest.from_file(APP_SCRIPT, default_timeout=BOOT_TIMEOUT)
    _seed_analyzed_project(at, analyzed_demo, trained_ols)
    at.run()

    assert not at.exception, f"App raised with seeded state: {at.exception}"

    radio_labels = [r.label for r in at.radio]
    assert any("Optimization Type" in label for label in radio_labels), (
        f"Expected optimizer radio to be rendered. Found radios: {radio_labels}"
    )


def test_single_objective_optimization_runs_end_to_end(analyzed_demo, trained_ols):
    """Full-path test:
      1. Seed a trained OLS model into session_state.
      2. Pick the model in the Optimizer tab selectbox.
      3. Click 'Run Single-Objective Optimization'.
      4. Assert the optimizer populated `single_opt_results` without error.
    """
    at = AppTest.from_file(APP_SCRIPT, default_timeout=BOOT_TIMEOUT)
    _seed_analyzed_project(at, analyzed_demo, trained_ols)
    at.run()
    assert not at.exception

    # The single-objective selectbox has key="single_opt_model".
    model_sb = at.selectbox(key="single_opt_model")
    assert len(model_sb.options) >= 1
    model_sb.select(model_sb.options[0]).run()
    assert not at.exception

    # Find the 'Run Single-Objective Optimization' button by label — it's the
    # one without a user-facing key= in optimizer_view.py.
    run_buttons = [
        b for b in at.button
        if "Run Single-Objective Optimization" in b.label
    ]
    assert run_buttons, (
        f"Run button not found. Visible buttons: {[b.label for b in at.button]}"
    )
    run_buttons[0].click().run()

    assert not at.exception, f"Optimizer raised: {at.exception}"

    # AppTest's session_state proxy supports __getitem__ / __contains__ but
    # not .get(), so we check-then-read.
    assert "single_opt_results" in at.session_state
    result = at.session_state["single_opt_results"]
    assert result is not None, "Optimizer did not populate single_opt_results"
    assert "dosages" in result and "outcome" in result
    assert len(result["dosages"]) == len(analyzed_demo["independent_vars"])
